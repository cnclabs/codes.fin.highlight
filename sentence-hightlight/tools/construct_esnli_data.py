import re
import os
import json
import string
import argparse
from spacy.lang.en import English
nlp = English()

def extract_marks_esnli(sentence):
    """
    This extraction method is different from fin10k's annotation,
    since esnli is used for training, need to tokenized by SpaCy
    """
    tokens = list()
    tokens_hl = list()
    labels = list()
    hl = 0
    punc = (lambda x: x in string.punctuation)

    for tok in nlp(sentence):
        if tok.text == "*":
            hl = 0 if hl else 1
        else:
            if punc(tok.text): 
                tokens_hl += []
                labels += [0]
            else:
                tokens_hl += [tok.text] if hl else []
                labels += [1] if hl else [0]
            tokens += [tok.text]

    return tokens_hl, labels, tokens

def convert_to_highlight(args):

    fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')

    highlight = 0
    with open(args.path_input_file, 'r') as fin:
        for i, line in enumerate(fin): 
            data_dict = json.loads(line.strip())

            tokensA_hl, labelsA, tokensA = extract_marks_esnli(data_dict['sentA_marked'].strip())
            tokensB_hl, labelsB, tokensB = extract_marks_esnli(data_dict['sentB_marked'].strip())

            nA, nB = len(tokensA), len(tokensB)
            n_words_parsed = len(data_dict['words']) - 3

            # only A
            labels = [-1] + [-100] * nA + [-1] + labelsB + [-1]
            probs = [-1] + [0] * nA + [-1] + labelsB + [-1]

            data_dict.update({
                'keywordsA': tokensA_hl, 'keywordsB': tokensB_hl,
                'labels': labels, 'probs': probs
            })

            del data_dict['sentA_marked']
            del data_dict['sentB_marked']

            highlight += len(tokensA_hl) 
            highlight += len(tokensB_hl) 

            if n_words_parsed != (nA+nB):
                print(f'Inconsistent un-marked tokens w/ sentences. {n_words_parsed}, {nA+nB}')
                print('This example will not be included in the training data of highlighting.')
            else:
                fout.write(json.dumps(data_dict) + '\n')

            if i % 1000 == 0:
                print(f"{i} annotation examples with {highlight} highlighted tokens.")

    fin.close()
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    args = parser.parse_args()

    # makedirs
    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)

    convert_to_highlight(args)

    print("Done")
