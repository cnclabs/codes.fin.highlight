import re
import json
import string
import spacy
import argparse
from nltk.corpus import stopwords
from collections import defaultdict
from spacy.lang.en import English
from utils import read_fin10K, load_master_dict, load_stopwords
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input_file", type=str)
parser.add_argument("-tokenizer", "--model_name", type=str, default='bert-base-uncased')
args = parser.parse_args()

if args.model_name == 'naive':
    fin = open(args.path_input_file, 'r')
    fout = open(args.path_input_file+'.naive_filtered', 'w')
if args.model_name == 'bert-base-uncased':
    fin = open(args.path_input_file+'.naive_filtered', 'r')
    fout = open(args.path_input_file+'.bert_filtered', 'w')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)


for line in fin:
    data = json.loads(line.strip())
    if len(data['sentA'].split()) <= 256 and len(data['sentB'].split()) <= 256:
        if args.model_name == 'naive':
            fout.write(json.dumps(data) + '\n')
        elif args.model_name == 'bert-base-uncased':
            if len(tokenizer(data['sentA']).tokens()) <= 256 and \
               len(tokenizer(data['sentB']).tokens()) <= 256:
                   fout.write(json.dumps(data) + '\n')
    
