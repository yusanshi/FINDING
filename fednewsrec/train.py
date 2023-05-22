import copy
import datetime
import importlib
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import (BatchSampler, DataLoader, RandomSampler,
                              SequentialSampler, Subset)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# Must be put ahead of `from fednewsrec...`
# to avoid the error `RuntimeError: context has already been set`
# https://github.com/tqdm/tqdm/issues/611
if __name__ == '__main__':
    set_start_method('spawn')

import fednewsrec.homoencrypt as homoencrypt
from fednewsrec.dataset import NewsDataset, TrainingBehaviorsDataset
from fednewsrec.model.general.trainer.centralized import CentralizedModel
from fednewsrec.model.general.trainer.federated_group import \
    FederatedGroupModel
from fednewsrec.shared import args, device, logger
from fednewsrec.utils import (EarlyStopping, dict2table, flatten,
                              infinite_sampler, time_since)


def train():
    if args.tensorboard_runs_dir is None:

        class DummyWriter:

            def add_scalar(*args, **kwargs):
                pass

        writer = DummyWriter()
    else:
        writer = SummaryWriter(log_dir=os.path.join(
            args.tensorboard_runs_dir,
            f'{args.model}-{args.dataset}',
            f"{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}",
        ))

    try:
        pretrained_word_embedding = torch.from_numpy(
            np.load(f'./data/{args.dataset}/pretrained_word_embedding.npy')
        ).float().contiguous()
    except FileNotFoundError:
        logger.warning('Pretrained word embedding not found')
        pretrained_word_embedding = None

    Model = getattr(importlib.import_module(f"fednewsrec.model.{args.model}"),
                    args.model)
    model = Model(pretrained_word_embedding).to(device)
    logger.info(model)

    model.init_backprop()

    start_time = time.time()
    loss_full = []
    early_stopping = EarlyStopping(patience=args.patience)
    best_checkpoint = None
    best_val_metrics = None
    if args.save_checkpoint:
        checkpoint_dir = os.path.join(args.checkpoint_dir,
                                      f'{args.model}-{args.dataset}')
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    news_dataset = NewsDataset(f'data/{args.dataset}/news.tsv')
    datasets = {}
    try:
        if isinstance(model, CentralizedModel):
            batch = 0
            for epoch in trange(args.num_epochs,
                                desc='Training epochs',
                                unit='epochs',
                                position=1):
                epoch_hash = epoch % args.max_training_dataset_cache_num
                if epoch_hash in datasets:
                    dataset = datasets[epoch_hash]
                else:
                    dataset = TrainingBehaviorsDataset(
                        f'data/{args.dataset}/train.tsv', len(news_dataset),
                        epoch_hash)
                    datasets[epoch_hash] = dataset

                # Use `sampler=BatchSampler(...)` to support batch indexing of dataset, which is faster
                dataloader = DataLoader(
                    dataset,
                    batch_size=None,
                    sampler=BatchSampler(
                        RandomSampler(dataset)
                        if args.shuffle else SequentialSampler(dataset),
                        batch_size=args.batch_size,
                        drop_last=False,
                    ))
                for minibatch in tqdm(dataloader,
                                      desc='Training batches',
                                      unit='batches',
                                      position=0,
                                      leave=False):
                    if args.dry_run:
                        continue

                    for k in minibatch.keys():
                        if k in [
                                'history', 'positive_candidates',
                                'negative_candidates'
                        ]:
                            minibatch[k] = news_dataset.news[minibatch[k]].to(
                                device)
                        else:
                            minibatch[k] = minibatch[k].to(device)

                    y_pred = model(minibatch, news_dataset.news_pattern)
                    loss = model.backward(y_pred)
                    loss_full.append(loss)

                    if batch % args.num_batches_show_loss == 0:
                        writer.add_scalar('Train/Loss', loss, batch)
                        logger.info(
                            f"Time {time_since(start_time)}, epoch {epoch}, batch {batch}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                        )
                    batch += 1

                if epoch % args.num_epochs_validate == 0:
                    metrics = model.evaluate('val')
                    for metric, value in metrics.items():
                        writer.add_scalar(f'Validation/{metric}', value, epoch)

                    logger.info(
                        f"Time {time_since(start_time)}, epoch {epoch}, metrics:\n{dict2table(metrics)}"
                    )

                    early_stop, get_better = early_stopping(-metrics['AUC'])
                    if early_stop:
                        logger.info('Early stop.')
                        break
                    elif get_better:
                        best_checkpoint = copy.deepcopy(model.state_dict())
                        best_val_metrics = copy.deepcopy(metrics)
                        if args.save_checkpoint:
                            torch.save(
                                model.state_dict(),
                                os.path.join(checkpoint_dir,
                                             f"ckpt-{epoch}.pt"))

        elif isinstance(model, FederatedGroupModel):
            dataset = TrainingBehaviorsDataset(
                f'data/{args.dataset}/train.tsv', len(news_dataset), 0)
            user2indexs = defaultdict(list)
            for i, user in enumerate(dataset.data['user'].tolist()):
                user2indexs[user].append(i)

            for round in trange(args.num_rounds,
                                desc='Training rounds',
                                unit='rounds',
                                position=1):

                if round != 0 and round % args.num_rounds_interpolate == 0:
                    model.interpolate(round)

                if round != 0 and round % args.num_rounds_cluster == 0:
                    # TODO test whether reclustering change evaluation results much
                    logger.info(f'Reclustering users: {model.cluster_users()}')

                round_hash = round % args.max_training_dataset_cache_num
                if round_hash in datasets:
                    dataset = datasets[round_hash]
                else:
                    dataset = TrainingBehaviorsDataset(
                        f'data/{args.dataset}/train.tsv', len(news_dataset),
                        round_hash)
                    datasets[round_hash] = dataset

                loss = 0
                gradients = []
                num_sample = 0

                for group, group_users in tqdm(model.group2users.items(),
                                               desc='Training groups',
                                               unit='groups',
                                               position=0,
                                               leave=False):

                    if args.save_gpu_memory:
                        model.to(device, group)

                    users = random.sample(
                        group_users,
                        math.ceil(args.num_users_per_round * len(group_users) /
                                  len(user2indexs)))

                    subset = Subset(
                        dataset,
                        flatten([user2indexs[user] for user in users]))
                    num_sample += len(subset)

                    minibatch = next(
                        iter(DataLoader(subset, batch_size=len(subset))))

                    if args.dry_run:
                        continue

                    for k in minibatch.keys():
                        if k in [
                                'history', 'positive_candidates',
                                'negative_candidates'
                        ]:
                            minibatch[k] = news_dataset.news[minibatch[k]].to(
                                device)
                        else:
                            minibatch[k] = minibatch[k].to(device)

                    y_pred = model.local_models[group](
                        minibatch, news_dataset.news_pattern)
                    loss_single, gradient_single = model.backward(
                        y_pred, group)

                    loss += loss_single * len(subset)
                    gradients.append((len(subset), gradient_single))

                    if args.save_gpu_memory:
                        model.to('cpu', group)

                gradient = []
                for i in range(len(list(model.parameters()))):
                    data = [x[1][i] for x in gradients]
                    weights = [x[0] for x in gradients]
                    weights = [x / sum(weights) for x in weights]

                    if not args.homomorphic_encryption:
                        gradient.append(
                            sum(x * y for x, y in zip(data, weights)))
                    else:
                        shape = data[0].shape
                        data = [x.numpy().flatten() for x in data]
                        data = homoencrypt.weighted_average(data, weights)
                        gradient.append(torch.from_numpy(data).view(shape))

                model.step(gradient)

                loss /= num_sample
                loss_full.append(loss)

                if round % args.num_rounds_show_loss == 0:
                    writer.add_scalar('Train/Loss', loss, round)
                    logger.info(
                        f"Time {time_since(start_time)}, round {round}, current loss {loss:.4f}, average loss {np.mean(loss_full):.4f}, latest average loss {np.mean(loss_full[-10:]):.4f}"
                    )

                if round != 0 and round % args.num_rounds_validate == 0:
                    metrics = model.evaluate('val')
                    for metric, value in metrics.items():
                        writer.add_scalar(f'Validation/{metric}', value, round)

                    logger.info(
                        f"Time {time_since(start_time)}, round {round}, metrics:\n{dict2table(metrics)}"
                    )

                    early_stop, get_better = early_stopping(-metrics['AUC'])
                    if early_stop:
                        logger.info('Early stop.')
                        break
                    elif get_better:
                        best_checkpoint = copy.deepcopy(model.state_dict())
                        best_val_metrics = copy.deepcopy(metrics)
                        if args.save_checkpoint:
                            torch.save(
                                model.state_dict(),
                                os.path.join(checkpoint_dir,
                                             f"ckpt-{round}.pt"))

        else:
            raise NotImplementedError

    except KeyboardInterrupt:
        logger.info('Stop in advance')

    if best_val_metrics is not None:
        logger.info(
            f'Best metrics on validation set:\n{dict2table(best_val_metrics)}')
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint)
    metrics = model.evaluate('test')
    logger.info(f'Metrics on test set:\n{dict2table(metrics)}')


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Training {args.model} on {args.dataset}')
    try:
        train()
    except RuntimeError as e:
        logger.error(f'Runtime Error: {e}')
