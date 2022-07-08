import os
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
parser.add_argument("-tokenizer", "--model_name", type=str, default='naive')
args = parser.parse_args()

if args.model_name == 'naive':
    os.rename(args.path_input_file, args.path_input_file+".bak")
    fin = open(args.path_input_file+".bak", 'r')
    fout = open(args.path_input_file, 'w')
    tokenizer = (lambda x: x.split())
if args.model_name == 'bert-base-uncased':
    os.rename(args.path_input_file, args.path_input_file+".bak")
    fin = open(args.path_input_file+".bak", 'r')
    fout = open(args.path_input_file, 'w')
    hf_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer = (lambda x: hf_tokenizer(x).tokens())

for line in fin:
    data = json.loads(line.strip())
    if len(data['sentA'].split()) <= 256 and len(data['sentB'].split()) <= 256:
        if len(tokenizer(data['sentA'])) <= 256 and len(tokenizer(data['sentB'])) <= 256:
               fout.write(json.dumps(data) + '\n')
    
