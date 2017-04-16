# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import pickle
import fire
import networkx as nx
from collections import Counter
from sklearn.linear_model import LinearRegression
from gensim.models.wrappers import FastText
from gensim.matutils import unitvec

import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def tokenize(text):
    wc = Counter()
    yield ('<s>', 0)
    for tok in text.split():
        yield (tok, wc[tok])
        wc[tok] += 1
    yield ('</s>', 0)


def add_edges(G, tokens):
    tokens = list(tokens)
    for x, y in zip(tokens, tokens[1:]):
        try:
            w = G.edge[x][y].get('weight', 0.0) + 1.0
            G.add_edges_from([(x, y, {'weight': w})])
        except KeyError:
            G.add_edges_from([(x, y, {'weight': 1.0})])
    return G


def get_vec(model, tokens):
    dim = model.wv.vector_size
    X = np.zeros((len(tokens), dim))
    for i, tok in enumerate(tokens):
        X[i] = model.wv.word_vec(tok)
    return X.mean(axis=0)


def main(ft_src, ft_tgt, tm_model, corpus, hyp_num=100):
    ft_src_model = FastText.load_fasttext_format(ft_src)
    ft_tgt_model = FastText.load_fasttext_format(ft_tgt)
    tm = pickle.load(open(tm_model, 'rb'))

    for i, line in enumerate(io.open(corpus, 'r', encoding='utf-8')):
        sent_pairs = line.strip().split(' ||| ')
        source = sent_pairs[0]
        targets = sent_pairs[1:]

        G = nx.DiGraph()
        for sent in targets:
            G = add_edges(G, tokenize(sent))

        source_vec = get_vec(ft_src_model, source)
        source_vec_proj = tm.predict(source_vec.reshape(1, -1))[0]
        source_vec_proj = unitvec(source_vec_proj)
        candidates = Counter()
        for p in nx.all_simple_paths(G, ('<s>', 0), ('</s>', 0)):
            target_tokens = [t[0] for t in p[1:-1]]
            target_vec = unitvec(get_vec(ft_tgt_model, target_tokens))
            sim = np.dot(source_vec_proj, target_vec)
            target_text = ' '.join(target_tokens)
            candidates[target_text] = sim

        for text, sim in candidates.most_common(hyp_num):
            print('{0} ||| {1} ||| Cosine={}'.format(i, text, sim))


if __name__ == '__main__':
    fire.Fire(main)
