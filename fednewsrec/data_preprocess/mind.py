import pandas as pd
import swifter
import numpy as np
import argparse
import csv

from os import path
from pathlib import Path
from nltk.tokenize import word_tokenize
from collections import defaultdict

from fednewsrec.parameters import parse_args

args, extra_args = parse_args()


def parse_news(
    source,
    target,
    news2int_path,
    category2int_path,
    word2int_path,
):
    news = [
        pd.read_table(
            x,
            header=None,
            usecols=range(5),
            names=['id', 'category', 'subcategory', 'title', 'abstract'],
        ) for x in source
    ]
    news = pd.concat(news)
    news.fillna('', inplace=True)
    news.drop_duplicates(subset=['id'], inplace=True, ignore_index=True)

    news2int = {x: i for i, x in enumerate(news.id, start=1)}

    category2int = {}
    word2int = {}
    word2frequency = defaultdict(int)
    for row in news.itertuples(index=False):
        if row.category not in category2int:
            category2int[row.category] = len(category2int) + 1
        if row.subcategory not in category2int:
            category2int[row.subcategory] = len(category2int) + 1

        for w in word_tokenize(row.title.lower()):
            word2frequency[w] += 1
        for w in word_tokenize(row.abstract.lower()):
            word2frequency[w] += 1

    for k, v in word2frequency.items():
        if v >= args.word_frequency_threshold:
            word2int[k] = len(word2int) + 1

    words_length = news.swifter.apply(
        lambda row: pd.Series([
            len(word_tokenize(row.title.lower())),
            len(word_tokenize(row.abstract.lower())),
        ],
                              index=['title_length', 'abstract_length']),
        axis=1)
    num_words_title = int(
        np.percentile(words_length.title_length.to_numpy(),
                      args.title_length_percentile))
    num_words_abstract = int(
        np.percentile(words_length.abstract_length.to_numpy(),
                      args.abstract_length_percentile))
    print(
        f'Title length: {num_words_title}, abstract length: {num_words_abstract}'
    )

    def parse_row(row):
        title_words = word_tokenize(row.title.lower())[:num_words_title]
        title_words = [
            word2int[w] if w in word2int else 0 for w in title_words
        ]
        title_words.extend([0] * (num_words_title - len(title_words)))

        abstract_words = word_tokenize(
            row.abstract.lower())[:num_words_abstract]
        abstract_words = [
            word2int[w] if w in word2int else 0 for w in abstract_words
        ]
        abstract_words.extend([0] * (num_words_abstract - len(abstract_words)))

        new_row = [
            news2int[row.id] if row.id in news2int else 0,
            category2int[row.category] if row.category in category2int else 0,
            category2int[row.subcategory]
            if row.subcategory in category2int else 0,
            title_words,
            abstract_words,
        ]

        return pd.Series(
            new_row,
            index=['id', 'category', 'subcategory', 'title', 'abstract'])

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


def parse_users(source, user2int_path):
    behaviors = [
        pd.read_table(
            x,
            header=None,
            usecols=[1],
            names=['user'],
        ) for x in source
    ]
    behaviors = pd.concat(behaviors)
    behaviors.drop_duplicates(inplace=True)
    user2int = {x: i for i, x in enumerate(behaviors.user, start=1)}
    pd.DataFrame(user2int.items(), columns=['user',
                                            'int']).to_csv(user2int_path,
                                                           sep='\t',
                                                           index=False)


def parse_behaviors(
    source,
    target,
    news2int_path,
    user2int_path,
):
    print(f"Parse {source}")
    news2int = dict(pd.read_table(news2int_path).to_numpy().tolist())
    user2int = dict(pd.read_table(user2int_path).to_numpy().tolist())

    behaviors = pd.read_table(source,
                              header=None,
                              usecols=[1, 3, 4],
                              names=['user', 'history', 'candidates'])
    behaviors.history.fillna('', inplace=True)

    def parse_row(row):
        history = [
            news2int[x] for x in row.history.split()[-args.num_history:]
        ]
        history_length = len(history)
        history = [0] * (args.num_history - history_length) + history
        new_row = [
            user2int[row.user],
            history,
            history_length,
            [
                news2int[x.split('-')[0]] for x in row.candidates.split()
                if x.endswith('-1')
            ],
            [
                news2int[x.split('-')[0]] for x in row.candidates.split()
                if x.endswith('-0')
            ],
        ]

        return pd.Series(new_row,
                         index=[
                             'user', 'history', 'history_length',
                             'positive_candidates', 'negative_candidates'
                         ])

    behaviors = behaviors.swifter.apply(parse_row, axis=1)
    behaviors.to_csv(target, sep='\t', index=False)


def generate_word_embedding(source, target, word2int_path):
    """
    Generate from pretrained word embedding file
    If a word not in embedding file, Initialize its embedding by N(0, 1)
    Args:
        source: path of pretrained word embedding file, e.g. glove.840B.300d.txt
        target: path for saving word embedding. Will be saved in numpy format
        word2int_path: vocabulary file when words in it will be searched in pretrained embedding file
    """
    # na_filter=False is needed since nan is also a valid word
    # word, int
    word2int = pd.read_table(word2int_path, na_filter=False, index_col='word')
    source_embedding = pd.read_table(source,
                                     index_col=0,
                                     sep=' ',
                                     header=None,
                                     quoting=csv.QUOTE_NONE,
                                     names=range(args.word_embedding_dim))
    # word, vector
    source_embedding.index.rename('word', inplace=True)
    # word, int, vector
    merged = word2int.merge(source_embedding,
                            how='inner',
                            left_index=True,
                            right_index=True)
    merged.set_index('int', inplace=True)

    missed_index = np.setdiff1d(np.arange(len(word2int) + 1),
                                merged.index.to_numpy())
    missed_embedding = pd.DataFrame(data=np.random.normal(
        size=(len(missed_index), args.word_embedding_dim)))
    missed_embedding['int'] = missed_index
    missed_embedding.set_index('int', inplace=True)

    final_embedding = pd.concat([merged, missed_embedding]).sort_index()
    np.save(target, final_embedding.to_numpy())

    print(
        f'Rate of word missed in pretrained embedding: {(len(missed_index)-1)/len(word2int):.4f}'
    )


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
    parser.add_argument(
        '--glove_path',
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

    print('Parse news')
    parse_news(
        [
            path.join(args.source_dir, x, 'news.tsv')
            for x in ['train', 'val', 'test']
        ],
        path.join(args.target_dir, 'news.tsv'),
        path.join(args.target_dir, 'news2int.tsv'),
        path.join(args.target_dir, 'category2int.tsv'),
        path.join(args.target_dir, 'word2int.tsv'),
    )

    print('Generate word embedding')
    generate_word_embedding(
        args.glove_path,
        path.join(args.target_dir, 'pretrained_word_embedding.npy'),
        path.join(args.target_dir, 'word2int.tsv'),
    )

    print('Parse users')
    parse_users(
        [
            path.join(args.source_dir, x, 'behaviors.tsv')
            for x in ['train', 'val', 'test']
        ],
        path.join(args.target_dir, 'user2int.tsv'),
    )

    print('Parse behaviors')
    for x in ['train', 'val', 'test']:
        parse_behaviors(
            path.join(args.source_dir, x, 'behaviors.tsv'),
            path.join(args.target_dir, f'{x}.tsv'),
            path.join(args.target_dir, 'news2int.tsv'),
            path.join(args.target_dir, 'user2int.tsv'),
        )
