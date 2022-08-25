import os
import json
import string
import spacy
import random
import argparse
import numpy as np
from collections import defaultdict
from spacy.lang.en import English
from utils import read_fin10k
from synthetic_utils import load_stopwords, load_master_dict

def heuristic_labeling(example, 
                       positive_threshold=0,
                       random_ratio=0,
                       negative_ratio=-1,):

    labelsA_pseudo, labelsB_pseudo = list(), list()
    keywordsA_pseudo, keywordsB_pseudo = list(), list()

    # POSITIVE/NEGATIVE list of tokens
    overlaps = [1 if tok in example['wordsA'] else 0 for tok in example['wordsB']]

    # Condtition functions
    punc = (lambda x: x in string.punctuation)
    stops = (lambda x: x in STOPWORDS)
    selfs = (lambda i: overlaps[i] == 1) 
    neighbors = (lambda i, x: ((overlaps+[1])[i+1] * ([1]+overlaps)[i]) == 1) 
    numbers = (lambda x: x.isdigit())

    # No label on sentence A
    labelsA_pseudo = [-100] * len(example['wordsA'])

    # Extract sentence B 
    for i, tok in enumerate(example['wordsB']):
        tokc = tok.casefold()
        if selfs(i):
            labelsB_pseudo += [0]
        elif stops(tokc) or punc(tokc):
            labelsB_pseudo += [0]
        elif numbers(tokc):
            if neighbors(i, tokc):
                labelsB_pseudo += [0]
            else:
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

    if negative_ratio > 1:
        negative_indices = [i for i, l in enumerate(labelsB_pseudo) if l == 0]
        n_relaxed = int(max(len(negative_indices) - negative_ratio*len(keywordsB_pseudo), 0))

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
    pass

def lexicon_based_labeling(example, 
                           positive_threshold=0,
                           random_ratio=0,
                           negative_ratio=-1,):

    labelsA_pseudo, labelsB_pseudo = list(), list()
    keywordsA_pseudo, keywordsB_pseudo = list(), list()

    # POSITIVE/NEGATIVE list of tokens
    overlaps = [1 if tok in example['wordsA'] else 0 for tok in example['wordsB']]

    # Condtition functions
    punc = (lambda x: x in string.punctuation)
    stops = (lambda x: x in STOPWORDS)
    finstops = (lambda x: x in FINSTOPWORDS)
    fins = (lambda x: x in FINWORDS)
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

    if negative_ratio > 1:
        negative_indices = [i for i, l in enumerate(labelsB_pseudo) if l == 0]
        n_relaxed = int(max(len(negative_indices) - negative_ratio*len(keywordsB_pseudo), 0))

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

def convert_to_highlight(args):

    fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')

    j, pos, neg = 0, 0, 0
    with open(args.path_input_file, 'r') as fin:
        for i, line in enumerate(fin): 
            data_dict = json.loads(line.strip())

            if args.synthetic_type == 'lexicon-based':
                flag = lexicon_based_labeling(
                    data_dict,
                    positive_threshold=args.n_hard_positive,
                    random_ratio=args.random_ratio,
                    negative_ratio=args.negative_sampling,
                )
            else:
                flag = heuristic_labeling(
                    data_dict,
                    positive_threshold=args.n_hard_positive,
                    random_ratio=args.random_ratio,
                    negative_ratio=args.negative_sampling,
                )

            # write the valid training synthtetic
            if flag:
                j += 1
                fout.write(json.dumps(data_dict) + '\n')
                pos += (np.array(data_dict['labels'])==1).sum()
                neg += (np.array(data_dict['labels'])==0).sum()

            if i % 5000 == 0:
                print(f"{i} / {j} synthetic examples with {pos} positve and {neg} negative.")

    fin.close()
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("-stopword", "--stopword_source", type=str, default='nltk')
    parser.add_argument("-synthetic", "--synthetic_type", type=str)
    parser.add_argument("-n_hard", "--n_hard_positive", default=None, type=int)
    parser.add_argument("-neg_sampling", "--negative_sampling", type=float, default=-1)
    parser.add_argument("-random", "--random_ratio", default=0, type=float)
    parser.add_argument("-spacy_sep", "--use_spacy_sep", action='store_true', default=False)
    # Lexcion based labeling
    parser.add_argument("-lexicon_sent", "--path_lexicon_sent_file", type=str, default=None)
    parser.add_argument("-lexicon_stop", "--path_lexicon_stop_file", type=str, default=None)
    args = parser.parse_args()

    random.seed(1234)
    # makedirs
    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)


    STOPWORDS = load_stopwords(args.stopword_source)

    if args.synthetic_type == 'heuristic':
        convert_to_highlight(args)

    if args.synthetic_type == 'lexicon-based':
        FINWORDS = load_master_dict(args.path_lexicon_sent_file)
        FINSTOPWORDS = load_stopwords(args.path_lexicon_stop_file)
        convert_to_highlight(args)

    print("Done")
