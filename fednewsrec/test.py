import numpy as np
import torch
import os
import sys
import importlib

from torch.multiprocessing import Process, SimpleQueue, set_start_method
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if __name__ == '__main__':
    set_start_method('spawn')

from fednewsrec.shared import args, logger, device
from fednewsrec.utils import latest_checkpoint, dict2table, calculate_cos_similarity, aggregate_metrics
from fednewsrec.dataset import NewsDataset, UserDataset, EvaluationBehaviorsDataset


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def calculate_single_user_metric(pair):
    try:
        auc = roc_auc_score(*pair)
        mrr = mrr_score(*pair)
        ndcg5 = ndcg_score(*pair, 5)
        ndcg10 = ndcg_score(*pair, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


def scoring_worker_fn(index,
                      task_queue,
                      mode,
                      news_vectors,
                      user_vectors,
                      prediction_fn,
                      selected_behaviors=None):
    if args.mute_fedgroup_evaluation:
        # Silence the subprocesses
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    behaviors_dataset = EvaluationBehaviorsDataset(
        f'data/{args.dataset}/{mode}.tsv', {}, index, args.num_scoring_workers)
    logger.debug(f'Scoring worker dataset size {len(behaviors_dataset)}')
    if selected_behaviors is not None:
        behaviors_dataset = Subset(behaviors_dataset, selected_behaviors)
        logger.debug(
            f'Scoring worker dataset size (after filtering) {len(behaviors_dataset)}'
        )

    for behaviors in behaviors_dataset:
        candidates = behaviors['positive_candidates'] + behaviors[
            'negative_candidates']
        news_vector = news_vectors[candidates]
        user_vector = user_vectors[behaviors['user_index']]
        click_probability = prediction_fn(news_vector, user_vector)

        y_pred = click_probability.tolist()
        y_true = [1] * len(behaviors['positive_candidates']) + [0] * len(
            behaviors['negative_candidates'])
        task_queue.put((y_true, y_pred))


def metrics_worker_fn(task_queue, result_queue):
    for task in iter(task_queue.get, None):
        result_queue.put(calculate_single_user_metric(task))


@torch.no_grad()
def infer_news_users(model, mode):
    model.eval()
    news_dataset = NewsDataset(f'data/{args.dataset}/news.tsv')
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 drop_last=False)
    news_vectors = []
    for minibatch in tqdm(news_dataloader,
                          desc='Calculating vectors for news'):
        news_vectors.append(
            model.get_news_vector(minibatch.to(device),
                                  news_dataset.news_pattern))
    news_vectors = torch.cat(news_vectors, dim=0)

    user_dataset = UserDataset(f'data/{args.dataset}/{mode}.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=args.batch_size * 16,
                                 shuffle=False,
                                 drop_last=False)

    user_vectors = []
    for minibatch in tqdm(user_dataloader,
                          desc='Calculating vectors for users'):
        user_vectors.append(
            model.get_user_vector(news_vectors[minibatch['history']]))

    user_vectors = torch.cat(user_vectors, dim=0)
    model.train()
    return news_vectors, user_dataset, user_vectors


@torch.no_grad()
def evaluate(model, mode, return_raw=False, selected_users=None):
    """
    Args:

    Returns:
        AUC
        MRR
        nDCG@5
        nDCG@10
    """
    assert mode in ['val', 'test']
    news_vectors, user_dataset, user_vectors = infer_news_users(model, mode)

    if args.show_similarity:
        logger.info(
            f"News cos similarity: {calculate_cos_similarity(news_vectors.cpu().numpy()[1:]):.4f}"
        )
        logger.info(
            f"User cos similarity: {calculate_cos_similarity(user_vectors.cpu().numpy()):.4f}"
        )

    behaviors_count = 0
    if selected_users is not None:
        total_selected_behaviors = []
        # If `i` is the index in `user_vectors`
        # then `user_data_index_to_id[i]` is its 'id' attribute value
        user_data_index_to_id = user_dataset.user['user'].tolist()

    for i in range(args.num_scoring_workers):
        # The part has an implicit effect: make sure the cache exists,
        # so in `scoring_worker_fn`, the `user2index` parameter
        # for `EvaluationBehaviorsDataset` can be empty.
        # In this way, `user_dataset.user2index` does not to be passed to `scoring_worker_fn`,
        # saving a lot time on pickling/unpickling
        behaviors_dataset = EvaluationBehaviorsDataset(
            f'data/{args.dataset}/{mode}.tsv', user_dataset.user2index, i,
            args.num_scoring_workers)
        if selected_users is None:
            behaviors_count += len(behaviors_dataset)
        else:
            # `i`: data index in a split behavior file
            # `user_index`: the index in `user_vectors`
            selected_behaviors = [
                i for i, user_index in enumerate(behaviors_dataset.user_index)
                if user_data_index_to_id[user_index] in selected_users
            ]

            total_selected_behaviors.append(selected_behaviors)
            behaviors_count += len(selected_behaviors)

    logger.debug(f'Number of behaviors to evaluate: {behaviors_count}')
    """
    Evaluation with multiprocessing:

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 0 │
                                                  └──────────────────┘

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 1 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 0 │
    └──────────────────┘                          ┌──────────────────┐
                                                  │ Metrics Worker 2 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 1 │       TASK QUEUE                                     RESULT QUEUE
    └──────────────────┘   ───────────────────►   ┌──────────────────┐     ───────────────────►
                                                  │ Metrics Worker 3 │
    ┌──────────────────┐                          └──────────────────┘
    │ Scoring Worker 2 │
    └──────────────────┘                          ┌──────────────────┐
                                                  │ Metrics Worker 4 │
                                                  └──────────────────┘

                                                  ┌──────────────────┐
                                                  │ Metrics Worker 5 │
                                                  └──────────────────┘
    """
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()

    scoring_workers = []
    for i in range(args.num_scoring_workers):
        worker = Process(target=scoring_worker_fn,
                         args=(i, task_queue, mode, news_vectors, user_vectors,
                               model.get_prediction,
                               total_selected_behaviors[i]
                               if selected_users is not None else None))
        worker.start()
        scoring_workers.append(worker)

    metrics_workers = []
    for _ in range(args.num_metrics_workers):
        worker = Process(target=metrics_worker_fn,
                         args=(task_queue, result_queue))
        worker.start()
        metrics_workers.append(worker)

    logger.debug('Scoring and metrics workers started')
    results = []
    with tqdm(total=behaviors_count,
              desc='Calculating metrics with multiprocessing') as pbar:
        while len(results) < behaviors_count:
            results.append(result_queue.get())
            pbar.update()

        logger.debug('Get all the results')

    for worker in scoring_workers:
        worker.join()
    logger.debug('All scoring workers joined')
    for _ in range(args.num_metrics_workers):
        task_queue.put(None)
    for worker in metrics_workers:
        worker.join()
    logger.debug('All metrics workers joined')
    if return_raw:
        return results

    return aggregate_metrics(results)


if __name__ == '__main__':
    logger.info(args)
    logger.info(f'Using device: {device}')
    logger.info(f'Testing {args.model} on {args.dataset}')
    Model = getattr(importlib.import_module(f"fednewsrec.model.{args.model}"),
                    args.model)
    model = Model().to(device)
    checkpoint_path = latest_checkpoint(
        os.path.join(args.checkpoint_dir, f'{args.model}-{args.dataset}'))
    if checkpoint_path is None:
        logger.warning(
            'No checkpoint file found! Evaluating with randomly initiated model'
        )
    else:
        logger.info(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    metrics = model.evaluate('test')
    logger.info(f'Metrics on test set:\n{dict2table(metrics)}')
