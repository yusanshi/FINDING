import importlib
from collections import defaultdict

import fednewsrec.homoencrypt as homoencrypt
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fednewsrec.shared import args, device, logger
from fednewsrec.test import evaluate, infer_news_users
from fednewsrec.utils import aggregate_metrics, flatten, map_dict, silenced

builtin_round = round

if args.mute_fedgroup_evaluation:
    # Reduce the noise ...
    evaluate = silenced(evaluate)


class Clusterer:

    def __init__(self, method, num_clusters):
        self.num_clusters = num_clusters
        if len(method) == 1:
            if method[0] == 'kmeans':
                self.clusterer = KMeans(n_clusters=num_clusters)
            elif method[0] == 'dbscan':
                # TODO the parameters for DBSCAN
                # TODO -1 (noise) just use the global model?
                raise NotImplementedError
            elif method[0] == 'hierarchical':
                raise NotImplementedError
            else:
                raise NotImplementedError
        elif len(method) == 3:
            raise NotImplementedError
        else:
            raise ValueError

    def fit_predict(self, features):
        if not args.homomorphic_encryption:
            return self.clusterer.fit_predict(features)
        else:
            return homoencrypt.kmeans(features, self.num_clusters)

    def predict(self, features):
        return self.clusterer.predict(features)


def create_optimizer(parameters):
    if args.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=args.learning_rate)
    else:
        raise NotImplementedError


def optimizer_parameters(optimizer):
    if isinstance(optimizer, torch.optim.Adam):
        # or `optimizer.state_dict()['state'].values()`
        for x in optimizer.state.values():
            yield x['exp_avg']
            yield x['exp_avg_sq']

    else:
        raise NotImplementedError


# Monkey-patching
torch.optim.Optimizer.parameters = optimizer_parameters


def maximize_diagonal(matrix):
    """
    Reorder the columns to maximize the sum of diagonal (trace).
    Return:
        A dict mapping old column index to new column index.
    """
    if matrix.shape[0] > matrix.shape[1]:
        matrix = matrix[:matrix.shape[1]]
    elif matrix.shape[0] < matrix.shape[1]:
        matrix = np.pad(matrix, [(0, matrix.shape[1] - matrix.shape[0]),
                                 (0, 0)],
                        mode='constant')
    _, column_indexs = linear_sum_assignment(matrix, maximize=True)
    return {x: i for i, x in enumerate(column_indexs)}


class FederatedGroupModel:

    def init_backprop(self):
        if args.loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

        self.optimizer = create_optimizer(self.parameters())

        Model = getattr(
            importlib.import_module(f"fednewsrec.model.{args.model}"),
            args.model)
        self.local_models = [
            Model().to(device) for _ in range(args.num_groups)
        ]
        for x in self.local_models:
            nn.Module.load_state_dict(x, nn.Module.state_dict(self))
            x.optimizer = create_optimizer(x.parameters())

        self.cluster_users(initial=True)

        current_length = 0
        self.prefix2depth = {}
        self.prefix2range = {}
        for name, parameter in self.named_parameters():
            prefix = '.'.join(
                name.split('.')[:args.parameter_interpolation_granularity])
            if prefix not in self.prefix2depth:
                self.prefix2depth[prefix] = len(self.prefix2depth)
                self.prefix2range[prefix] = [
                    current_length, current_length + parameter.numel()
                ]
            else:
                assert list(self.prefix2depth.keys())[-1] == prefix
                self.prefix2range[prefix][1] += parameter.numel()
            current_length += parameter.numel()

        logger.info(
            f'Layer depths: {self.prefix2depth}, layer lengths: {self.prefix2range}'
        )

        if isinstance(self.optimizer, torch.optim.Adam):
            self.prefix2optimizer_range = {
                k: (2 * v[0], 2 * v[1])
                for k, v in self.prefix2range.items()
            }
        else:
            raise NotImplementedError

    def backward(self, y_pred, index):
        if args.loss == 'BCE':
            y = torch.cat(
                (torch.ones(y_pred.size(0), 1),
                 torch.zeros(y_pred.size(0), args.negative_sampling_ratio)),
                dim=1).to(device)
        elif args.loss == 'CE':
            y = torch.zeros(y_pred.size(0)).long().to(device)
        else:
            raise NotImplementedError

        loss = self.criterion(y_pred, y)
        self.local_models[index].optimizer.zero_grad()
        loss.backward()
        self.local_models[index].optimizer.step()
        return loss.item(), [
            x.grad for x in self.local_models[index].parameters()
        ]

    def step(self, gradient):
        """
        Step for the global model
        """
        for x, y in zip(self.parameters(), gradient):
            x.grad = y
        self.optimizer.step()

    @torch.no_grad()
    def _get_local_parameters(self):
        model_parameters = torch.stack(
            [parameters_to_vector(x.parameters()) for x in self.local_models],
            dim=0)
        optimizer_parameters = torch.stack([
            parameters_to_vector(x.optimizer.parameters())
            for x in self.local_models
        ],
                                           dim=0)
        return model_parameters, optimizer_parameters

    @torch.no_grad()
    def _set_local_parameters(self, model_parameters, optimizer_parameters):
        for x, y, z in zip(self.local_models, model_parameters,
                           optimizer_parameters):
            vector_to_parameters(y, x.parameters())
            # TODO what if skip reinitializing optmizers? (by comment out the following lines)
            # x.optimizer = create_optimizer(x.parameters())
            # x.optimizer.load_state_dict(self.optimizer.state_dict(
            # ))  # Need this or `x.optimizer.state` return nothing
            vector_to_parameters(z, x.optimizer.parameters())

    @torch.no_grad()
    def interpolate(self, round):
        local_model_parameters, local_optimizer_parameters = self._get_local_parameters(
        )
        global_model_parameters = parameters_to_vector(self.parameters())
        global_optimizer_parameters = parameters_to_vector(
            self.optimizer.parameters())
        p_all = {}
        for prefix, depth in self.prefix2depth.items():
            range = self.prefix2range[prefix]
            optimizer_range = self.prefix2optimizer_range[prefix]

            p = args.personalization_coefficient(r=round,
                                                 i=depth,
                                                 n=len(self.prefix2depth))
            p_all[depth] = builtin_round(p, 4)

            local_model_parameters[:, range[0]:range[
                1]] = local_model_parameters[:, range[0]:range[
                    1]] * p + global_model_parameters[range[0]:range[1]] * (1 -
                                                                            p)
            local_optimizer_parameters[:, optimizer_range[0]:optimizer_range[
                1]] = local_optimizer_parameters[:, optimizer_range[
                    0]:optimizer_range[1]] * p + global_optimizer_parameters[
                        optimizer_range[0]:optimizer_range[1]] * (1 - p)

        if round % 50 == 0:
            logger.info(
                f'Personalization coefficient {p_all} on round {round}')
        self._set_local_parameters(local_model_parameters,
                                   local_optimizer_parameters)

    @torch.no_grad()
    def cluster_users(self, initial=False):
        _, user_dataset, user_vectors = infer_news_users(self, 'train')
        # user data id (indexs in `user_vectors`) to real id
        user_id_map = dict(enumerate(user_dataset.user['user'].tolist()))
        user_vectors = user_vectors.cpu().numpy()
        self.clusterer = Clusterer(args.clustering_method, args.num_groups)
        groups = self.clusterer.fit_predict(user_vectors)

        if not initial:
            # Only interpolate the model on groups changes
            old_map = self.groups
            new_map = groups

            def get_transfer_matrix(old, new):
                return torch.sparse_coo_tensor(
                    np.array([old, new]), torch.ones(len(old),
                                                     dtype=torch.long),
                    (old.max() + 1, new.max() + 1)).to_dense()

            transfer_matrix = get_transfer_matrix(old_map, new_map).numpy()
            group_map = maximize_diagonal(transfer_matrix)
            self.group_map = group_map
            groups = new_map = np.vectorize(group_map.get)(new_map)
            transfer_matrix = get_transfer_matrix(old_map, new_map)

            transfer_matrix = (transfer_matrix /
                               transfer_matrix.sum(dim=0)).t()
            transfer_matrix = transfer_matrix.to(device)

            old_model_parameters, old_optimizer_parameters = self._get_local_parameters(
            )
            new_model_parameters = torch.mm(transfer_matrix,
                                            old_model_parameters)
            new_optimizer_parameters = torch.mm(transfer_matrix,
                                                old_optimizer_parameters)
            self._set_local_parameters(new_model_parameters,
                                       new_optimizer_parameters)

        self.groups = groups
        self.user2group = {
            user_id_map[user]: group
            for user, group in enumerate(groups)
        }
        self.group2users = defaultdict(list)
        for user, group in self.user2group.items():
            self.group2users[group].append(user)

    @torch.no_grad()
    def evaluate(self, mode):
        if not hasattr(self, 'local_models'):
            logger.warning(
                'Local models not initiated! Evaluating with the global model')
            return evaluate(self, mode)

        # TODO the users without history?

        _, user_dataset, user_vectors = infer_news_users(self, mode)

        # user data id (indexs in `user_vectors`) to real id
        user_id_map = dict(enumerate(user_dataset.user['user'].tolist()))
        user_vectors = user_vectors.cpu().numpy()
        groups = self.clusterer.predict(user_vectors)
        try:
            groups = np.vectorize(self.group_map.get)(groups)
        except AttributeError:
            pass  # don't need a remap
        user2group = {
            user_id_map[user]: group
            for user, group in enumerate(groups)
        }
        group2users = defaultdict(list)
        for user, group in user2group.items():
            group2users[group].append(user)

        local_results = {}
        for i, x in enumerate(self.local_models):
            logger.info(
                f'Evaluating users for group {i} (size: {len(group2users[i])})'
            )
            if len(group2users[i]) == 0:
                logger.warning(f'Group {i} has no users, skip')
                continue

            local_results[i] = evaluate(x,
                                        mode,
                                        return_raw=True,
                                        selected_users=set(group2users[i]))

        local_metrics = aggregate_metrics(flatten(local_results.values()))
        return local_metrics

    def to(self, device, index=None, with_optimizer=True):
        if index is None:
            # Move the global model
            return nn.Module.to(self, device)
        else:
            # Move the local model
            nn.Module.to(self.local_models[index], device, non_blocking=True)
            if with_optimizer:
                for x in self.local_models[index].optimizer.state.values():
                    for v in x.values():
                        if isinstance(v, torch.Tensor):
                            v.data = v.data.to(device, non_blocking=True)

    def state_dict(self):
        return {
            'global': nn.Module.state_dict(self),
            'local': [nn.Module.state_dict(x) for x in self.local_models],
            'clusterer': self.clusterer,
            'group_map': self.group_map
        }

    def load_state_dict(self, checkpoint):
        nn.Module.load_state_dict(self, checkpoint['global'])
        for x, y in zip(self.local_models, checkpoint['local']):
            nn.Module.load_state_dict(x, y)
        self.clusterer = checkpoint['clusterer']
        self.group_map = checkpoint['group_map']
        logger.warning(
            'The model is broken after loading the checkpoint, use it only for evaluation'
        )
