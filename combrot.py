# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import pickle
import fire
import numpy as np
import networkx as nx
from collections import Counter

import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def add_edges(G, tokens):
    tokens = list(tokens)
    for x, y in zip(tokens, tokens[1:]):
        try:
            w = G.edge[x][y].get('weight', 0.0) + 1.0
            G.add_edges_from([(x, y, {'weight': w})])
        except KeyError:
            G.add_edges_from([(x, y, {'weight': 1.0})])
    return G


def path_cost(G, path, weight='weight'):
    cost = 0
    for i in range(len(path)):
        if i > 0:
            if weight != None:
                edge = (path[i - 1], path[i])
                cost += G.get_edge_data(*edge)[weight]
            else:
                cost += 1
    if cost == 0:
        return 0
    return cost / len(path)


def get_vec(model, tokens):
    dim = model.wv.vector_size
    X = np.zeros((len(tokens), dim))
    for i, tok in enumerate(tokens):
        X[i] = model.wv.word_vec(tok)
    return X.mean(axis=0)


def get_longest_common_subseq(data):
    substr = []
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0]) - i + 1):
                if j > len(substr) and is_subseq_of_any(data[0][i:i + j],
                                                        data):
                    substr = data[0][i:i + j]
    return substr


def is_subseq_of_any(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if not is_subseq(find, data[i]):
            return False
    return True


# Will also return True if possible_subseq == seq.
def is_subseq(possible_subseq, seq):
    if len(possible_subseq) > len(seq):
        return False

    def get_length_n_slices(n):
        for i in range(len(seq) + 1 - n):
            yield seq[i:i + n]

    for slyce in get_length_n_slices(len(possible_subseq)):
        if slyce == possible_subseq:
            return True
    return False


def merge_subseq(targets):
    phrases = {}
    while True:
        long_subseq = get_longest_common_subseq(targets)
        if len(long_subseq) == 1:
            break
        key = 'p' + str(len(phrases) + 1)
        phrases[key] = ' '.join(long_subseq)
        for i, t in enumerate(targets):
            t = ' '.join(t)
            lsub = ' '.join(long_subseq)
            targets[i] = t.replace(lsub, key).split()

    for i, tokens in enumerate(targets):
        targets[i] = [phrases.get(t, t) for t in tokens]

    return targets


def handle_tokens(tokens):
    wc = Counter()
    yield ('<s>', 0)
    for tok in tokens:
        yield (tok, wc[tok])
        wc[tok] += 1
    yield ('</s>', 0)


def main(corpus, hyp_num=1000):
    for i, line in enumerate(io.open(corpus, 'r', encoding='utf-8')):
        sent_pairs = line.strip().split(' ||| ')
        targets = merge_subseq([tok.split() for tok in sent_pairs[1:]])

        G = nx.DiGraph()
        for tokens in targets:
            G = add_edges(G, handle_tokens(tokens))

        for j, p in enumerate(sorted(
                nx.all_simple_paths(G, ('<s>', 0), ('</s>', 0)),
                key=lambda path: path_cost(G, path, weight='weight'))):
            if j >= hyp_num:
                break
            target_tokens = [t[0] for t in p[1:-1]]
            target_text = ' '.join(target_tokens)
            print('{0} ||| {1}'.format(i, target_text))


if __name__ == '__main__':
    fire.Fire(main)
