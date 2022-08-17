import os
import json
import string
import spacy
import random
import argparse
import numpy as np
from nltk.corpus import stopwords
from collections import defaultdict
from spacy.lang.en import English
from utils import read_fin10k, load_master_dict, load_stopwords, extract_marks

def heuristic_labeling():
    pass
def lexicon_based_labeling(args, 
                           example, 
                           onlyB=True, 
                           positive_threshold=0,
                           random_ratio=0,
                           negative_sampling=False,
                           stopword_removal=True,
                           version=4):

    labelsA_pseudo, labelsB_pseudo = list(), list()
    keywordsA_pseudo, keywordsB_pseudo = list(), list()

    # POSITIVE/NEGATIVE list of tokens
    finwords = load_master_dict(args.path_lexicon_sent_file)
    overlaps = [1 if tok in example['wordsA'] else 0 for tok in example['wordsB']]
    stopwords = load_stopwords(args.stopword_source)
    finstopwords = load_stopwords(source=args.path_lexicon_stop_file)

    # Condtition functions
    punc = (lambda x: x in string.punctuation)
    stops = (lambda x: x in stopwords)
    finstops = (lambda x: x in finstopwords)
    fins = (lambda x: x in finwords)
    selfs = (lambda i: overlaps[i] == 1) 
    # True if index i was appeared
    neighbors = (lambda i, x: ((overlaps+[1])[i+1] * ([1]+overlaps)[i]) == 1) 
    # True if index i's neighbors were appered
    numbers = (lambda x: x.isdigit())
    # times = (lambda x: x in list(map(str, range(2010, 2018))))

    # No label on sentence A
    labelsA_pseudo = [-100] * len(example['wordsA'])

    # Extract sentence B (fs_fin10k_v1)
    for i, tok in enumerate(example['wordsB']):
        tokc = tok.casefold()
        if selfs(i):
            labelsB_pseudo += [0]
        elif stops(tokc) or punc(tokc) or finstops(tokc):
            labelsB_pseudo += [0]
        elif numbers(tokc):
            if neighbors(i, tokc):
                labelsB_pseudo += [0]
            else:
                labelsB_pseudo += [1]
                keywordsB_pseudo += [tok]
        elif fins(tokc):
            labelsB_pseudo += [1]
            keywordsB_pseudo += [tok]
        elif neighbors(i, tokc):
            labelsB_pseudo += [1]
            keywordsB_pseudo += [tok]
        else:
            if random_ratio >= random.uniform(0, 1):
                labelsB_pseudo += [1]
                keywordsB_pseudo += [tok]
            else:
                labelsB_pseudo += [0]

    if negative_sampling:
        negative_indices = [i for i, l in enumerate(labelsB_pseudo) if l == 0]
        # 1:1 sampling (relax (n - n_pos))
        n_relaxed = max(len(negative_indices) - len(keywordsB_pseudo), 0)

        for relaxed_index in random.sample(negative_indices, n_relaxed):
            labelsB_pseudo[relaxed_index] = -100

    example.update({
            'keywordsA': keywordsA_pseudo,
            'keywordsB': keywordsB_pseudo,
            'labels': labelsA_pseudo + labelsB_pseudo
    })

    if positive_threshold:
        return True if len(keywordsB_pseudo) > positive_threshold else False
    else:
        return True

def convert_to_bert_synthetic(args):
    nlp = English()

    fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')

    j = 0
    pos, neg = 0, 0
    with open(args.path_input_file, 'r') as fin:

        for line in fin: 
            data = json.load(line.strip())

            flag = lexicon_based_labeling(
                args, example,
                onlyB=not args.labeling_on_sentA,
                positive_threshold=args.n_hard_positive,
                stopword_removal=True,
                random_ratio=args.random_ratio,
                negative_sampling=args.negative_sampling,
                version=args.version
            )
        if flag:
            f.write(json.dumps(example) + '\n')
            j += 1
            # counting label distribution
            pos += (np.array(example['labels'])==1).sum()
            neg += (np.array(example['labels'])==0).sum()

        if i % 5000 == 0:
            print(f"{i} / {j} synthetic examples with {pos} positve and {neg} negative.")

# def convert_to_bert(args):
#     nlp = English()
#     data = read_fin10k(args.path_input_file, True)
#
#     f = open(args.path_output_file, 'w')
#     for i, example in enumerate(data):
#         example_token = token_extraction(
#                 example['sentA'], example['sentB'],
#                 pair_type=args.fin10k_type,
#                 spacy_sep=args.use_spacy_sep,
#         )
#         example.update(example_token)
#
#         # other pair information
#         if args.output_format == 'jsonl':
#             f.write(json.dumps(example) + '\n')
#         elif args.output_format == 'tsv':
#             f.write(
#                 f"{example['idA']}#{example['sentA']}\
#                     \n{example['idB']}#{example['sentB']}\
#                     \nTokensB#{'#'.join(example['wordsB'])}\
#                     \nAnnotation-1#{'#'.join([str(l) for l in example['labels']])}\
#                     \nAnnotation-2#{'#'.join([str(0) for l in example['labels']])}\
#                     \nAnnotation-3#{'#'.join([str(0) for l in example['labels']])}\n"
#             )
#         else:
#             exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("-type", "--fin10k_type", type=int, default=-1)
    # For synthetic data
    parser.add_argument("-lexicon_sent", "--path_lexicon_sent_file", type=str)
    parser.add_argument("-lexicon_stop", "--path_lexicon_stop_file", type=str, default=None)
    parser.add_argument("-stopword", "--stopword_source", type=str, default='nltk')
    parser.add_argument("-synthetic", "--synthetic_type", type=str)
    parser.add_argument("-version", "--version", default=4, type=int)
    parser.add_argument("-n_hard", "--n_hard_positive", default=None, type=int)
    parser.add_argument("-random", "--random_ratio", default=0, type=float)
    parser.add_argument("-neg_sampling", "--negative_sampling", action='store_true', default=False)
    parser.add_argument("-spacy_sep", "--use_spacy_sep", action='store_true', default=False)
    args = parser.parse_args()
    nlp = English()

    random.seed(1234)
    # makedirs
    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)

    if args.synthetic_type == 'heuristic':
        convert_to_bert_synthetic(args)

    if args.synthetic_type == 'lexicon-based':
        convert_to_bert_synthetic(args)

    print("Done")
