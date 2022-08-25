import re
import os
import json
import argparse

def extract_marks(tokens):

    labels = []
    extracted = []
    p_marks = re.compile(r"[\*].*?[\*]")
    # if not isinstance(tokens, list):
    #     exit(0)

    for i, tok in enumerate(tokens):
        marked = p_marks.findall(tok)
        if len(marked) > 0:
            tokens[i] = tok.replace("*", "")
            extracted += [tokens[i]]
            labels.append(1)
        else:
            labels.append(0)

    return extracted, labels

def convert_raw_to_highlight(args):

    fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')

    highlight = 0
    with open(args.path_input_file, 'r') as fin:
        for i, line in enumerate(fin): 
            data_dict = json.loads(line.strip())

            srcA = data_dict['sentA'].replace("*", "")
            srcB = data_dict['sentB'].replace("*", "")
            words = [tok.replace("*", "") for tok in data_dict['words']]

            tokensA_hl, labelsA = extract_marks(data_dict['wordsA'])
            tokensB_hl, labelsB = extract_marks(data_dict['wordsB'])

            labels = [-1] + labelsA + [-1] + labelsB + [-1]
            probs = [-1] + labelsA + [-1] + labelsB + [-1]

            data_dict.update({
                'sentA': srcA, 'sentB': srcB,
                'words': words,
                'keywordsA': tokensA_hl, 'keywordsB': tokensB_hl,
                'labels': labels, 'probs': probs
            })
            highlight += len(tokensA_hl) 
            highlight += len(tokensB_hl) 

            fout.write(json.dumps(data_dict) + '\n')

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

    convert_raw_to_highlight(args)

    print("Done")
