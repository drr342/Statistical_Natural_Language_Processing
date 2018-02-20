# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:44:29 2017

@author: drr34
"""

import gensim, logging, argparse
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('files', nargs = 2)
args = parser.parse_args()

wv = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(args.files[0], binary = True)
wv.save_word2vec_format(args.files[1] + '.wv')