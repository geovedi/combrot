# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import io
import pickle
import fire
import numpy as np
from sklearn.linear_model import LinearRegression
from gensim.models.wrappers import FastText
from gensim.matutils import unitvec

import logging
logging.basicConfig(
    format='%(asctime)s [%(process)d] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)


def line_count(fname):
    for i, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
        pass
    return i + 1


def get_vec(model, fname):
    dim = model.wv.vector_size
    vec = np.zeros((line_count(fname), dim))
    for i, line in enumerate(io.open(fname, 'r', encoding='utf-8')):
        tokens = line.strip().split()
        X = np.zeros((len(tokens), dim))
        for i, tok in enumerate(tokens):
            X[i] = model.wv.word_vec(tok)
        vec[i] = X.mean(axis=0)
    return np.asarray(vec)


def main(ft_src, ft_tgt, corpus_src, corpus_tgt, out_fname):
    ft_src_model = FastText.load_fasttext_format(ft_src)
    ft_tgt_model = FastText.load_fasttext_format(ft_tgt)

    X = get_vec(ft_src_model, corpus_src)
    y = get_vec(ft_tgt_model, corpus_tgt)
    assert X.shape == y.shape, 'mismatched shapes'

    lr = LinearRegression()
    lr.fit(X, y)

    with io.open(out_fname, 'wb') as out:
        pickle.dump(lr, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(main)

