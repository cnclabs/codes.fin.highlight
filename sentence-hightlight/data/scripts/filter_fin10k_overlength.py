import os
import re
import json
import string
import spacy
import argparse
from nltk.corpus import stopwords
from collections import defaultdict
from spacy.lang.en import English
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input_file", type=str)
parser.add_argument("-ol_out", "--path_overlength_file", type=str, default='overlength.txt')
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
    lengthA = len(data['sentA'].split())
    lengthB = len(data['sentA'].split())
    if lengthA <= 512 and lengthB <= 512:
        if len(tokenizer(data['sentB'])) <= 509:
           fout.write(json.dumps(data) + '\n')
    else:
        print("Over-length sentence pair found: {} and {}".format(
             len(data['sentA'].split()), len(data['sentB'].split())
        ))
        with open(args.path_overlength_file, 'a') as fol:
            fol.write(json.dumps(data) + '\n')

fin.close()
fout.close()


    
