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

def load_stopwords(source, max_num=None):
    stopwords_list = []
    if source == 'nltk':
        from nltk.corpus import stopwords
        stopwords_list = [w.casefold() for w in stopwords.words('english')]
    elif source == 'anserini':
        with open("stopwords_en.txt", 'r') as f:
            for line in f:
                if "#" not in line.strip():
                    stopwords_list += [line.strip()]
    elif source is not None:
        with open(source, 'r') as f:
            stopwords_list = [line.strip() for line in f.readlines()]

    return [w.casefold() for w in stopwords_list][:max_num]


def read_fin10K(path):
    """ Function for reading the sentence A/B from parsed financial 10K report."""
    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            try:
                idA, idB, sentA, sentB = line.strip().split('\t')
            except:
                print("************* ERROR FORMATTING *************")
                print(line.strip().split('\t'))
                print("************* ERROR FORMATTING *************")

            data['idA'].append(idA)
            data['sentA'].append(sentA)
            data['idB'].append(idB)
            data['sentB'].append(sentB)

            # score = list(map(float, scores.strip().split()))
            # data['scores'].append( (scores) )

    return data

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

    return extracted


