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
        lengthA = count_tokens(data['sentA'])
        lengthB = count_tokens(data['sentB'])

        # [Add first one to accelerate]
        if args.is_train and (lengthA + lengthB) < 508:
           fout.write(json.dumps(data) + '\n')
        elif args.is_train is False and lengthB < 400:
           fout.write(json.dumps(data) + '\n')

        else:
            print(f"Over-length sentence pair found (untokenized): {lengthA} and {lengthB}")
            if args.path_overlength_file is None:
                path_ol_output = args.path_input_file.replace("type2", "type1")
            else:
                path_ol_output = args.path_overlength_file
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
    parser.add_argument("-out_ol", "--path_overlength_file", type=str, default=None)
    parser.add_argument("-tokenizer", "--tokenizer_name", type=str, default='bert-base-uncased')
    parser.add_argument("-is_train", "--is_train", action='store_true', default=False)
    args = parser.parse_args()

    filtering(args)
