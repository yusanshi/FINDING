import pandas as pd
import swifter
import numpy as np
import argparse
import json
import random

import os
from os import path
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections import defaultdict
from tqdm import tqdm

from fednewsrec.parameters import parse_args

args, extra_args = parse_args()


def clean_data(source_dir, source_dir_clean):
    Path(source_dir_clean).mkdir(parents=True, exist_ok=True)

    required_keys = ['id', 'title', 'userId', 'time']
    optional_keys = ['category1']
    all_keys = required_keys + optional_keys
    for file in os.listdir(source_dir):
        data = []
        with open(path.join(source_dir, file)) as f:
            for line in tqdm(f):
                line = json.loads(line)
                if all(key in line for key in required_keys):
                    data.append({
                        key: line[key] if key in line else None
                        for key in all_keys
                    })

        df = pd.DataFrame(data)
        df.sort_values('time', ascending=True, inplace=True)
        df.drop('time', axis=1, inplace=True)
        df.to_csv(path.join(source_dir_clean, f'{file}.tsv'),
                  sep='\t',
                  index=False)


def parse_news(
    df,
    target,
    news2int_path,
    category2int_path,
    word2int_path,
):
    news = df[['id', 'title', 'category1']]
    assert len(news.drop_duplicates()) == len(
        news.drop_duplicates(subset=['id']))

    news = news.drop_duplicates(ignore_index=True)
    news2int = {x: i for i, x in enumerate(news.id, start=1)}

    category2int = {}
    word2int = {}
    word2frequency = defaultdict(int)
    for row in news.itertuples(index=False):
        if row.category1 is not np.nan:
            category, subcategory = row.category1.split('|')
            if category not in category2int:
                category2int[category] = len(category2int) + 1
            if subcategory not in category2int:
                category2int[subcategory] = len(category2int) + 1

        for w in word_tokenize(row.title.lower(), language='norwegian'):
            word2frequency[w] += 1

    for k, v in word2frequency.items():
        if v >= args.word_frequency_threshold:
            word2int[k] = len(word2int) + 1

    title_length = news.swifter.apply(lambda row: len(
        word_tokenize(row.title.lower(), language='norwegian')),
                                      axis=1)
    num_words_title = int(
        np.percentile(title_length.to_numpy(), args.title_length_percentile))
    print(f'Title length: {num_words_title}')

    def parse_row(row):
        title_words = word_tokenize(row.title.lower(),
                                    language='norwegian')[:num_words_title]
        title_words = [
            word2int[w] if w in word2int else 0 for w in title_words
        ]
        title_words.extend([0] * (num_words_title - len(title_words)))

        if row.category1 is not np.nan:
            category, subcategory = row.category1.split('|')
        else:
            category, subcategory = None, None

        new_row = [
            news2int[row.id] if row.id in news2int else 0,
            category2int[category] if category in category2int else 0,
            category2int[subcategory] if subcategory in category2int else 0,
            title_words,
        ]

        return pd.Series(new_row,
                         index=['id', 'category', 'subcategory', 'title'])

    news = news.swifter.apply(parse_row, axis=1)
    news.to_csv(target, sep='\t', index=False)

    pd.DataFrame(news2int.items(), columns=['news',
                                            'int']).to_csv(news2int_path,
                                                           sep='\t',
                                                           index=False)
    pd.DataFrame(category2int.items(),
                 columns=['category', 'int']).to_csv(category2int_path,
                                                     sep='\t',
                                                     index=False)

    pd.DataFrame(word2int.items(), columns=['word',
                                            'int']).to_csv(word2int_path,
                                                           sep='\t',
                                                           index=False)


def parse_users(dataframes, user2int_path):
    dataframes = [dataframes[k] for k in sorted(dataframes)]
    train_df = pd.concat(dataframes[args.history_days:args.history_days +
                                    args.train_days])
    val_test_df = pd.concat(dataframes[args.history_days + args.train_days:])
    users = sorted(list(set(train_df.userId) & set(val_test_df.userId)))
    user2int = {x: i for i, x in enumerate(users, start=1)}
    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)


def parse_behaviors(
    dataframes,
    target_dir,
    news2int_path,
    user2int_path,
):
    news2int = dict(pd.read_table(news2int_path).to_numpy().tolist())
    user2int = dict(pd.read_table(user2int_path).to_numpy().tolist())

    dataframes = [
        pd.DataFrame({
            'user': dataframes[k]['userId'].map(user2int),
            'news': dataframes[k]['id'].map(news2int)
        }).dropna().astype(int) for k in sorted(dataframes)
    ]

    assert len(
        dataframes) == args.history_days + args.train_days + args.val_test_days
    train_history_df = pd.concat(dataframes[:args.history_days])
    val_test_history_df = pd.concat(dataframes[:args.history_days +
                                               args.train_days])

    train_df = pd.concat(dataframes[args.history_days:args.history_days +
                                    args.train_days])
    val_test_df = pd.concat(dataframes[args.history_days + args.train_days:])
    val_df = val_test_df.sample(frac=args.val_ratio)
    test_df = val_test_df.drop(val_df.index)

    df_combined = pd.concat(dataframes)
    user2positive = df_combined.groupby('user')['news'].agg(set).to_dict()

    def parse(behaviors, history_df, mode):
        print(f"Parse {mode}")
        user2history = history_df.groupby('user')['news'].agg(list).to_dict()
        behaviors = behaviors.groupby('user').agg(list).reset_index()

        def parse_row(row):
            history = user2history[
                row.
                user][-args.num_history:] if row.user in user2history else []

            history_length = len(history)
            history = [0] * (args.num_history - history_length) + history

            negative_candidates = []
            positives = user2positive[row.user]
            while len(negative_candidates
                      ) < args.negative_candidates_per_click * len(row.news):
                negative = random.randrange(len(news2int) + 1)
                if negative not in positives and negative not in negative_candidates:
                    negative_candidates.append(negative)

            new_row = [
                row.user,
                history,
                history_length,
                row.news,
                negative_candidates,
            ]

            return pd.Series(new_row,
                             index=[
                                 'user', 'history', 'history_length',
                                 'positive_candidates', 'negative_candidates'
                             ])

        behaviors = behaviors.swifter.apply(parse_row, axis=1)
        # TODO drop this?
        behaviors.drop(behaviors[behaviors.history_length == 0].index,
                       inplace=True)
        behaviors.to_csv(path.join(target_dir, f'{mode}.tsv'),
                         sep='\t',
                         index=False)

    parse(train_df, train_history_df, 'train')
    parse(val_df, val_test_history_df, 'val')
    parse(test_df, val_test_history_df, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_dir',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--target_dir',
        type=str,
        required=True,
    )

    # Merge local and global args
    args_local, extra_args_local = parser.parse_known_args()
    for k, v in args_local.__dict__.items():
        args.__dict__[k] = v

    for x in extra_args + extra_args_local:
        if x.startswith('--'):
            x = x[2:].split('=')[0]
            assert x in args, f'Unknown args: {x}'

    Path(args.target_dir).mkdir(parents=True, exist_ok=True)

    args.val_ratio = 0.2
    args.negative_candidates_per_click = 20

    # TODO
    assert args.source_dir.endswith('adressa-1week')
    args.history_days = 5
    args.train_days = 1
    args.val_test_days = 1

    args.source_dir_clean = str(
        Path(args.source_dir).parent / (Path(args.source_dir).stem + '-clean'))

    print('Clean raw data')
    clean_data(args.source_dir, args.source_dir_clean)

    dataframes = {
        file: pd.read_table(path.join(args.source_dir_clean, file))
        for file in os.listdir(args.source_dir_clean)
    }
    df_combined = pd.concat(dataframes.values())

    print('Parse news')
    parse_news(
        df_combined,
        path.join(args.target_dir, 'news.tsv'),
        path.join(args.target_dir, 'news2int.tsv'),
        path.join(args.target_dir, 'category2int.tsv'),
        path.join(args.target_dir, 'word2int.tsv'),
    )

    print('Parse users')
    parse_users(
        dataframes,
        path.join(args.target_dir, 'user2int.tsv'),
    )

    print('Parse behaviors')
    parse_behaviors(
        dataframes,
        args.target_dir,
        path.join(args.target_dir, 'news2int.tsv'),
        path.join(args.target_dir, 'user2int.tsv'),
    )
