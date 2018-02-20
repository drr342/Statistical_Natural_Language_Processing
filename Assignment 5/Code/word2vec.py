# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:38:46 2017

@author: drr34
"""

import gensim, logging, os, argparse
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-size', type = int, default = 100)
parser.add_argument('-sg', type = int, default = 0)
parser.add_argument('-hs', type = int, default = 0)
parser.add_argument('-iter', type = int, default = 5)
parser.add_argument('-window', type = int, default = 5)
parser.add_argument('-extra', action = 'store_true')
parser.add_argument('files', nargs = 2)
args = parser.parse_args()

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if (not args.extra and fname[-3:] != '.1m'):
                continue
            for line in open(os.path.join(self.dirname, fname), encoding="utf8"):
                yield line.split()
 
sentences = MySentences(args.files[0]) # a memory-friendly iterator

model = gensim.models.Word2Vec(sentences, size = args.size, sg = args.sg, hs = args.hs,
                               window = args.window, iter = args.iter)
model.wv.save_word2vec_format(args.files[1] + '.wv')

