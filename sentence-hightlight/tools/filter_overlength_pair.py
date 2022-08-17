import os
import re
import json
import spacy
import argparse
from nltk.corpus import stopwords
from collections import defaultdict
from spacy.lang.en import English
from transformers import AutoTokenizer

def filtering(args):

    n_ol = 0

    # data prepare
    os.rename(args.path_input_file, args.path_input_file+".bak")
    fin = open(args.path_input_file+".bak", 'r')
    fout = open(args.path_input_file, 'w')

    # tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    count_tokens = (lambda x: len(hf_tokenizer(x).tokens()))

    for line in fin:
        data = json.loads(line.strip())
        lengthA = len(data['sentA'].split())
        lengthB = len(data['sentB'].split())

        # [Add first one to accelerate]
        if lengthB <= 400 and count_tokens(data['sentB']) <= 500:
               fout.write(json.dumps(data) + '\n')
        else:
            print(f"Over-length sentence pair found (untokenized): {lengthA} and {lengthB}")
            path_ol_output = args.path_input_file.replace("type2", "type1")
            with open(path_ol_output, 'a') as fol:
                data['type'] = 1
                fol.write(json.dumps(data) + '\n')
            n_ol +=1

    fin.close()
    fout.close()
    
    print(f"Total number of overlength parir: {n_ol}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--path_input_file", type=str)
    parser.add_argument("-tokenizer", "--tokenizer_name", type=str, default='bert-base-uncased')
    args = parser.parse_args()

    filtering(args)
