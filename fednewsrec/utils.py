import time
import os
import logging
import coloredlogs
import datetime
import hashlib
import pickle
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import sys
import copy

from pathlib import Path
from torch.multiprocessing import current_process
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm
from contextlib import redirect_stdout, redirect_stderr


def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def create_logger(args):
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=args.logging_level,
                        logger=logger,
                        fmt='%(asctime)s %(levelname)s %(message)s')

    class TqdmLoggingHandler(logging.Handler):

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except Exception:
                self.handleError(record)

    # Method from `tqdm.contrib.logging.logging_redirect_tqdm`
    assert len(logger.handlers) == 1
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logger.handlers[0].formatter)
    logger.handlers = [tqdm_handler]

    if current_process().name == 'MainProcess':
        # Skip logging to file in child processes
        log_dir = os.path.join(args.log_dir, f'{args.model}-{args.dataset}')
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(
            log_dir,
            f"{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}.txt"
        )
        logger.info(f'Check {log_file_path} for the log of this run')
        file_handler = logging.FileHandler(log_file_path)
        logger.addHandler(file_handler)

    class ExitingHandler(logging.Handler):

        def emit(self, record):
            if record.levelno >= logging.ERROR:
                sys.exit(-1)

    logger.addHandler(ExitingHandler())
    return logger


def round_float(x):
    # Precision for abs(x):
    # [0, 0.01)     6
    # [0.01, 0.1)   5
    # [0.1, 1)      4
    # [1, 10)       3
    # [10, 100)     2
    # [100, oo)     1
    precision = max(1, min(4 - math.ceil(math.log(abs(x), 10)), 6))
    return f'{x:.{precision}f}'


def map_dict(d, fn=round_float):
    '''
    Map a nested dict with function `fn`
    '''

    def _map_dict(d, fn):
        for k, v in d.items():
            if isinstance(v, dict):
                _map_dict(v, fn)
            else:
                d[k] = fn(v)

    d = copy.deepcopy(d)
    _map_dict(d, fn)
    return d


def dict2table(d, k_fn=str, v_fn=round_float, corner_name=''):
    '''
    Convert a nested dict to markdown table
    '''

    def parse_header(d, depth=0):
        assert depth in [0, 1], 'Only 1d or 2d dicts allowed'
        if isinstance(list(d.values())[0], dict):
            header = parse_header(list(d.values())[0], depth=depth + 1)
            for v in d.values():
                assert header == parse_header(v, depth=depth + 1)
            return header
        else:
            return f"| {' | '.join([corner_name] * depth + list(map(k_fn, d.keys())))} |"

    def parse_segmentation(d):
        return ' --- '.join(['|'] * parse_header(d).count('|'))

    def parse_content(d, accumulated_keys=[]):
        if isinstance(list(d.values())[0], dict):
            contents = []
            for k, v in d.items():
                contents.extend(parse_content(v, accumulated_keys + [k_fn(k)]))
            return contents
        else:
            return [
                f"| {' | '.join(accumulated_keys + list(map(v_fn, d.values())))} |"
            ]

    lines = [parse_header(d), parse_segmentation(d), *parse_content(d)]
    return '\n'.join(lines)


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])


def load_from_cache(
        identifiers,
        generator,
        cache_dir,
        enabled,
        load_cache_callback=lambda x: print(f'Load cache from {x}'),
        save_cache_callback=lambda x: print(f'Save cache to {x}'),
        print=print):

    if not enabled:
        return generator()

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f"{hashlib.md5('-'.join(map(str,identifiers)).encode('utf-8')).hexdigest()}.pkl"
    )
    cache_lock_path = cache_path + '.lock'

    def load_cache():
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
            load_cache_callback(cache_path)
            return data

    if os.path.isfile(cache_lock_path):
        # some other is generating the cache
        # wait and simply load it
        while True:
            print(f'Waiting for the other program (lock: {cache_lock_path})')
            time.sleep(2)
            if not os.path.isfile(cache_lock_path):
                assert os.path.isfile(cache_path)
                return load_cache()
    else:
        if os.path.isfile(cache_path):
            return load_cache()

        # acquire the lock
        open(cache_lock_path, 'w').close(
        )  # TODO release the lock (delete the file) if exit unexpectedly (use with?)
        # generate the cache
        data = generator()
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        # release the lock
        os.remove(cache_lock_path)

        save_cache_callback(cache_path)
        return data


class EarlyStopping:

    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def calculate_cos_similarity(array, filename=None):
    indexs = list(range(len(array)))
    random.shuffle(indexs)
    for i, x in enumerate(indexs):
        if i == x:
            indexs[i] = (indexs[i] + 1) % len(array)
    another_array = array[indexs]
    array = array / np.linalg.norm(array, axis=1, keepdims=True)
    another_array = another_array / np.linalg.norm(
        another_array, axis=1, keepdims=True)
    data = (array * another_array).sum(axis=1)
    plt.hist(data, density=True, range=(0, 1), bins=500)
    plt.ylabel('Probability')
    plt.xlabel('Cos-sim')
    plt.title('Cos-sim distribution')
    if filename is not None:
        plt.savefig(filename, dpi=144)
    plt.close()
    return np.mean(data)


# https://stackoverflow.com/a/28321717/8418049
def silenced(func):

    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                return func(*args, **kwargs)

    return wrapper


def aggregate_metrics(results):
    return dict(
        zip(['AUC', 'MRR', 'nDCG@5', 'nDCG@10'],
            np.nanmean(np.array(results), axis=0)))


def flatten(list):
    """
    Flatten a 2d list
    https://stackoverflow.com/a/29244327/8418049
    """
    return [item for sub_list in list for item in sub_list]


def infinite_sampler(data,
                     batch_size,
                     shuffle=True,
                     drop_last=True,
                     replacement=False):
    """
    >>> s = infinite_sampler(range(10), 4, drop_last=False)
    >>> for _ in range(6):
    ...     print(next(s))
    [9, 8, 1, 5]
    [0, 2, 4, 7]
    [6, 3]
    [1, 5, 3, 9]
    [2, 7, 8, 4]
    [0, 6]
    """
    data = list(data)

    if replacement:
        assert shuffle
        while True:
            yield random.sample(data, batch_size)

    sampler = BatchSampler(
        RandomSampler(data) if shuffle else SequentialSampler(data),
        batch_size=batch_size,
        drop_last=drop_last,
    )

    while True:
        yield from sampler
