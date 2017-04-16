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


def get_vec(model, fname):
    vec = []
    for line in io.open(fname, 'r', encoding='utf-8'):
        tokens = line.strip().split()
        X = np.zeros((len(tokens), model.size))
        for i, tok in enumerate(tokens):
            X[i] = model.wv.word_vec(tok)
        vec.append(X.mean(axis=0))
    return np.asarray(vec)


def main(ft_src, ft_tgt, corpus_src, corpus_tgt, out_fname):
    ft_src_model = FastText.load_fasttext_format(ft_src)
    ft_tgt_model = FastText.load_fasttext_format(ft_tgt)

    X = get_vec(ft_src_model, corpus_src)
    y = get_vec(ft_tgt_model, corpus_tgt)

    lr = LinearRegression()
    lr.fit(X, y)

    with io.open(out_fname, 'wb') as out:
        pickle.dump(lr, out, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(main)

