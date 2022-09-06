import collections
from spacy.lang.en import English
import re
import json
import numpy as np

def read_esnli(path, class_selected=('contradiction', 'neutral', 'entailment')):
    """
    Returns:
        data_sorted: dictionary of sent pair information of a `List`.
        class_selected: string of selected class (one of above 3 classes)
    """

    # train files are split into two, need to merge before used
    import pandas as pd
    df = pd.read_csv(path)
    df.reset_index(inplace=True)
    df = df.rename(columns={
        'Sentence1': 'sentA', 'Sentence2': 'sentB',
        'Sentence1_marked_1': 'sentA_marked', 'Sentence2_marked_1': 'sentB_marked'
    })
    df = df.loc[:, ['pairID', 'gold_label', 'sentA', 'sentB', 'sentA_marked', 'sentB_marked']]
    data = collections.defaultdict(list)

    # Extract the e-snli data
    for index, row in df.iterrows():

        example = row.to_dict()
        if (example.pop('gold_label') in class_selected) and (row.isna().sum() == 0):
            data[example.pop('pairID')] = example

    # sanity check
    n_pairs = len(data)
    print(f"Total number of example: {n_pairs}")

    # sort by idb if using evaluation ser
    data_sorted = [v for k, v in sorted(data.items(), key=lambda x: x[0])]
    return data_sorted

def read_fin10k(path):
    """
    Returns:
        data_sorted: dictionary of sent pair information of a `List`.
    """
    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            idA, idB, sentA, sentB = line.strip().split('\t')

            data[f'{idA}#{idB}'] = {
                    "idB": idB,
                    "idA": idA,
                    "sentA": sentA,
                    "sentB": sentB
            }

    # sanity check
    n_pairs = len(data)
    n_sentB = len(set([id_pair.split('#')[1] for id_pair in data.keys()]))

    print(f"Total number of example: {n_pairs}, Total number of sentB: {n_sentB} ")

    # sort by idb if using evaluation ser
    data_sorted = [v for k, v in sorted(data.items(), key=lambda x: x[0])]
    return data_sorted

def token_extraction(srcA, srcB, pair_type=2, spacy_sep=False):
    """
    Args:
        srcA, srcB: Strings of sentence pair.
        spacy_sep: separate by using `spacy tokenizer`, used when creating train data.
    """
    # normalized strings
    srcA = re.sub("\s\s+" , " ", srcA)
    srcB = re.sub("\s\s+" , " ", srcB)

    if spacy_sep:
        nlp = English()
        tokensA = [tok.text for tok in nlp(srcA)]
        tokensB = [tok.text for tok in nlp(srcB)]
    else:
        tokensA = srcA.split()
        tokensB = srcB.split()

    labelsA = [0] * len(tokensA)
    probsA = labelsA
    # we only focus on highlight sentB
    if pair_type == 2:
        labelsB = [1] * len(tokensB)
    else:
        labelsB = [pair_type] * len(tokensB)
    probsB = labelsB

    return {'type': int(pair_type),
            'sentA': srcA, 'sentB': srcB,
            'words': ["<tag1>"] + tokensA + ["<tag2>"] + tokensB + ["<tag3>"],
            'wordsA': tokensA, 'wordsB': tokensB,
            'keywordsA': [], 'keywordsB': [],
            'labels': [-1] + labelsA + [-1] + labelsB + [-1],
            'probs': [-1] + probsA + [-1] + probsB + [-1],}

def load_truth(file_path, sentA=True):
    truth = {}

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            try:
                pair_id = data['idA'] + "#" + data['idB']
            except:
                pair_id = i

            truth[pair_id] = {
                    "keywords": data['keywordsB'], 
                    "text_pair": f"# {data['sentA']}\n# {data['sentB']}"
            }

    return truth

def load_pred(file_path, special_token=False, prob_threshold=0):
    prediction = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            try:
                pair_id = data.pop('idA') + "#" + data.pop('idB')
            except:
                pair_id = i

            prediction[pair_id] = []

            flag = False
            for j, (w, p) in enumerate(zip(data['words'], data['probs'])):

                if p == -1:
                    # when aggregation
                    if special_token:
                        prediction[pair_id].append( (w, p) )
                        flag = True
                    # when evalaution
                    else:
                        flag = False if j == 0 else True
                elif flag:
                    if p >= prob_threshold:
                        prediction[pair_id].append( (w, p) )
                    else:
                        prediction[pair_id].append( (w, 0) )
                else:
                    pass

    return prediction
