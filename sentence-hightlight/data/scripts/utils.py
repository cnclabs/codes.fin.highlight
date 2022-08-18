import re
import os
import json
import collections

def load_master_dict(path, max_num=None):
    lexicon = collections.defaultdict(int)
    with open(path, 'r') as f:
        for line in f:
            keyword, word_cnt, doc_cnt = line.strip().split('\t')
            lexicon[keyword] = doc_cnt

    return [w.casefold() for w in lexicon][:max_num]


def read_fin10k(path, is_eval=False):
    """ function for reading the sentence a/b from parsed financial 10k report."""
    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            idA, idB, sentA, sentB = line.strip().split('\t')

            if is_eval:
                data[f'{idA}#{idB}'] = {
                        "idA": idA, 
                        "idB": idB, 
                        "sentA": sentA, 
                        "sentB": sentB
                }
            else:
                data[f'{idA}#{idB}'] = {
                        "sentA": sentA, 
                        "sentB": sentB
                }

    print(f"Total number of exampels: {len(data)}")

    # sort by idb if using evaluation ser
    if is_eval:
        data_sorted = [v for k, v in sorted(data.items(), key=lambda x: x[0])]
        return data_sorted
    else:
        data_unsorted = [v for k, v in data.items()]
        return data_unsorted

def read_esnli(path, class_selected, reverse):
    """ Function for reading the sentence A/B and highlight A/B with the corresponding labels """
    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for i, item_dict in enumerate(f):
            items = json.loads(item_dict.strip())
            data['sentA'].append(items['Sentence1'])
            data['sentB'].append(items['Sentence2'])
            data['highlightA'].append(items['Marked1'])
            data['highlightB'].append(items['Marked2'])
            data['label'].append(items['label'])

    # example filtering 
    if class_selected != 'all':
        data['sentA'] = [h for (h, l) in zip(data['sentA'], data['label']) if l in class_selected]
        data['sentB'] = [h for (h, l) in zip(data['sentB'], data['label']) if l in class_selected]
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) if l in class_selected]
        data['highlightB'] = [h for (h, l) in zip(data['highlightB'], data['label']) if l in class_selected]
        data['label'] = [l for l in data['label'] if l in class_selected]

    if reverse:
        data['sentA'] = data['sentA'] + data['sentB']
        data['sentB'] = data['sentB'] + data['sentA']
        data['highlightA'] = data['highligthA'] + data['hightlightB']
        data['highlightB'] = data['highligthB'] + data['hightlightA']
        data['label'] = data['label'] + data['label']

    return data

def extract_marks(tokens):

    labels = []
    extracted = []
    p_marks = re.compile(r"[\*].*?[\*]")
    # p_punct = re.compile("[" + re.escape(string.punctuation) + "]")
    if not isinstance(tokens, list):
        exit(0)

    for i, tok in enumerate(tokens):
        marked = p_marks.findall(tok)
        if len(marked) > 0:
            # extracted += [p_punct.sub("", tok)]
            tokens[i] = tok.replace("*", "")
            extracted += [tokens[i]]
            labels.append(1)
        else:
            labels.append(0)

    return extracted, labels


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

    if pair_type == 2:
        labelsA = [0] * len(tokens_A)
        labelsB = [1] * len(tokens_B)
        probsA, probsB = labelsA, labelsB
    else:
        labelsA = [pair_type] * len(tokens_A) 
        labelsB = [pair_type] * len(tokens_B)
        probsA, probsB = labelsA, labelsB

    return {'type': int(pair_type),
            'sentA': srcA,
            'sentB': srcB,
            'words': ["<tag1>"] + tokens_A + ["<tag2>"] + tokens_B + ["<tag3>"],
            'wordsA': tokens_A,
            'wordsB': tokens_B,
            'keywordsA': tokens_A_hl,
            'keywordsB': tokens_B_hl,
            'labels': [-1] + labelsA + [-1] + labelsB + [-1],
            'probs': [-1] + probsA + [-1] + probsB + [-1],}

