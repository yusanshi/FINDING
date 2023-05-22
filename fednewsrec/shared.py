import pandas as pd
import torch
from torch.multiprocessing import current_process

from fednewsrec.parameters import parse_args
from fednewsrec.utils import create_logger

args, extra_args = parse_args()
logger = create_logger(args)

if len(extra_args) > 0:
    logger.error(f'Unknown args: {extra_args}')

if args.num_words is None:
    args.num_words = len(
        pd.read_table(f'data/{args.dataset}/word2int.tsv')) + 1
if args.num_categories is None:
    args.num_categories = len(
        pd.read_table(f'data/{args.dataset}/category2int.tsv')) + 1
if args.num_users is None:
    args.num_users = len(
        pd.read_table(f'data/{args.dataset}/user2int.tsv')) + 1
if args.dataset.startswith(
        'adressa-') and 'abstract' in args.dataset_attributes['news']:
    if current_process().name == 'MainProcess':
        logger.warning(
            'The Adressa dataset has no "abstract" attribute, remove it')
    args.dataset_attributes['news'].remove('abstract')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
