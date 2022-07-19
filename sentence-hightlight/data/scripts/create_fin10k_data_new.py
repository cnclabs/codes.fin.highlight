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
from utils import read_fin10k, read_fin10k_with_window, load_master_dict, load_stopwords, extract_marks

def token_extraction(srcA, srcB, pair_type=2, fully_seperated=False, marks_annotation=False):
    if fully_seperated:
        tokens_A, tokens_B = list(), list()
        for tok in nlp(srcA):
            tokens_A += [tok.text]
        for tok in nlp(srcB):
            tokens_B += [tok.text]
    else:
        tokens_A = srcA.split(' ')
        tokens_B = srcB.split(' ')

    if marks_annotation:
        srcA = srcA.replace("*", "")
        srcB = srcB.replace("*", "")
        tokens_A_hl, labelsA = extract_marks(tokens_A)
        tokens_B_hl, labelsB = extract_marks(tokens_B)
    else:
        tokens_A_hl, tokens_B_hl = [], []
        labelsA, labelsB = [], []

        if pair_type != 2:
            labelsA = [-1] + [pair_type] * len(tokens_A) + [-1] 
            labelsB = [-1] + [pair_type] * len(tokens_B) + [-1]

    return {'type': pair_type,
            'sentA': srcA,
            'sentB': srcB,
            'words': ["<tag1>"] + tokens_A + ["<tag2>"] + tokens_B + ["<tag3>"],
            'wordsA': tokens_A,
            'wordsB': tokens_B,
            'keywordsA': tokens_A_hl,
            'keywordsB': tokens_B_hl,
            'labels': labelsA,
            'prob': labelsB}

def lexicon_based_labeling(args, 
                           example, 
                           onlyB=True, 
                           positive_threshold=0,
                           random_ratio=0,
                           negative_sampling=False,
                           stopword_removal=True,
                           version=4):

    """ Convert the sentence pairs into the extracted tokens, which are ready to predict.
    [*] Stopwords, and label them as 0
    [*] Financial stopwords, and label them as 0
    [*] Label the sentimental words as 1
    [*] The other tokens that no referenced, labeled as -100 (ignore)
    """
    pseudo_labels_A, pseudo_labels_B = list(), list()
    tokens_A_hl, tokens_B_hl = list(), list()

    # POSITIVE
    ## lexicon (medium/weak postive)
    finwords = load_master_dict(args.path_lexicon_sent_file)
    ## continuous (hard positive)
    overlaps = [1 if tok in example['wordsA'] else 0 for tok in example['wordsB']]
    # overlaps = [example['wordsA'].index(tok) if tok in example['wordsA'] \
    #         else 0 for tok in example['wordsB']]
    # neighbors = (lambda i, x: (example['wordsA']+[" "])[overlaps[i-1]+1] == x)

    # NEGATIVE
    ## standard stopword (hard negative)
    stopwords = load_stopwords(args.stopword_source)
    ## standard financial stopwords (medium negative)
    finstopwords = load_stopwords(source=args.path_lexicon_stop_file)

    punc = (lambda x: x in string.punctuation)
    stops = (lambda x: x in stopwords)
    finstops = (lambda x: x in finstopwords)
    fins = (lambda x: x in finwords)
    selfs = (lambda i: overlaps[i] == 1) # if index i appeard in sentA
    neighbors = (lambda i, x: ((overlaps+[1])[i+1] * ([1]+overlaps)[i]) == 1) # if index i's neighbors are appered
    numbers = (lambda x: x.isdigit())
    times = (lambda x: x in list(map(str, range(2010, 2018))))

    # Extract sentence A
    for i, tok in enumerate(example['wordsA']):
        tokc = tok.casefold()
        if onlyB:
            tokens_A_hl = []
            pseudo_labels_A += [-100] 
        else:
            if stops(tokc) or punc(tokc) or finstops(tokc):
                pseudo_labels_A += [0]
            elif fins(tokc):
                tokens_A_hl += [tok]
                pseudo_labels_A += [1]
            else:
                pseudo_labels_A += [-100]


    # Extract sentence B (fs_fin10k_v1)
    for i, tok in enumerate(example['wordsB']):
        tokc = tok.casefold()
        if selfs(i):
            pseudo_labels_B += [0]
        elif stops(tokc) or punc(tokc) or finstops(tokc):
            pseudo_labels_B += [0]
        elif numbers(tokc):
            if neighbors(i, tokc):
                pseudo_labels_B += [0]
            else:
                pseudo_labels_B += [1]
                tokens_B_hl += [tok]
        elif fins(tokc):
            pseudo_labels_B += [1]
            tokens_B_hl += [tok]
        elif not neighbors(i, tokc):
            pseudo_labels_B += [1]
            tokens_B_hl += [tok]
        else:
            if random_ratio >= random.uniform(0, 1):
                pseudo_labels_B += [1]
                tokens_B_hl += [tok]
            else:
                pseudo_labels_B += [0]

    if negative_sampling:
        negative_indices = [i for i, l in enumerate(pseudo_labels_B) if l == 0]
        # 1:1 sampling (relax (n - n_pos))
        n_relaxed = max(len(negative_indices) - len(tokens_B_hl), 0)

        for relaxed_index in random.sample(negative_indices, n_relaxed):
            pseudo_labels_B[relaxed_index] = -100

    example.update({
            'keywordsA': tokens_A_hl,
            'keywordsB': tokens_B_hl,
            'labels': pseudo_labels_A + pseudo_labels_B
    })

    if positive_threshold:
        return True if len(tokens_B_hl) > positive_threshold else False
    else:
        return True

def convert_to_bert_synthetic(args):
    nlp = English()
    is_eval = ("eval" in args.path_input_file)
    if args.merge_global_window:
            data = read_fin10k_with_window(args.path_input_file, is_eval)
    else:
        data = read_fin10k(args.path_input_file, is_eval)

    f = open(args.path_output_file, 'w')
    j = 0
    pos, neg = 0, 0
    # for i, (sa, sb) in enumerate(zip(data['sentA'], data['sentB'])):
    for i, example in enumerate(data):
        example_token = token_extraction(
                example['sentA'], 
                example['sentB'], 
                fully_seperated=args.no_seperation
        )
        example.update(example_token)

        # synthetic label prepreocessing
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

def convert_to_bert(args):
    nlp = English()
    if args.merge_global_window:
        data = read_fin10k_with_window(args.path_input_file, True)
    else:
        data = read_fin10k(args.path_input_file, True)

    f = open(args.path_output_file, 'w')
    for i, example in enumerate(data):
        example_token = token_extraction(
                example['sentA'], 
                example['sentB'],
                pair_type=args.fin10k_type,
                fully_seperated=args.no_seperation,
                marks_annotation=args.is_truth
        )
        example.update(example_token)

        # other pair information
        if args.output_format == 'jsonl':
            f.write(json.dumps(example) + '\n')
        elif args.output_format == 'tsv':
            f.write(
                f"{example['idA']}#{example['sentA']}\
                    \n{example['idB']}#{example['sentB']}\
                    \nTokensB#{'#'.join(example['wordsB'])}\
                    \nAnnotation-1#{'#'.join([str(l) for l in example['labels']])}\
                    \nAnnotation-2#{'#'.join([str(0) for l in example['labels']])}\
                    \nAnnotation-3#{'#'.join([str(0) for l in example['labels']])}\n"
            )
        else:
            exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-highlight_A", "--labeling_on_sentA", action='store_true', default=False)
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("-format", "--output_format", type=str, default='jsonl')
    parser.add_argument("-model_type", "--model_type", type=str)
    parser.add_argument("-type", "--fin10k_type", type=int, default=-1)
    # For synthetic data
    parser.add_argument("-lexicon_sent", "--path_lexicon_sent_file", type=str)
    parser.add_argument("-lexicon_stop", "--path_lexicon_stop_file", type=str, default=None)
    parser.add_argument("-stopword", "--stopword_source", type=str, default='nltk')
    parser.add_argument("-version", "--version", default=4, type=int)
    parser.add_argument("-n_hard", "--n_hard_positive", default=None, type=int)
    parser.add_argument("-random", "--random_ratio", default=0, type=float)
    parser.add_argument("-neg_sampling", "--negative_sampling", action='store_true', default=False)
    # empirical
    parser.add_argument("-nosep", "--no_seperation", action='store_false', default=True)
    parser.add_argument("-annotation", "--is_truth", action='store_true', default=False)
    parser.add_argument("-global", "--merge_global_window", action='store_true', default=False)
    args = parser.parse_args()
    nlp = English()

    random.seed(1234)
    # makedirs
    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)

    if args.model_type == 'bert':
        convert_to_bert(args)
    elif args.model_type == 'synthetic':
        convert_to_bert_synthetic(args)

    print("Done")
