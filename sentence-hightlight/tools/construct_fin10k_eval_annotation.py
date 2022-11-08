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

    if args.output_csv:
        fout_csv = open(args.path_output_file.replace('jsonl', 'tsv'), 'w')

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
            start_of_b = data_dict['labels'][1:].index(-1) + 1
            labelsB = data_dict['labels'][(start_of_b+1):-1]

            print(f"{i} annotation examples with {highlight} highlighted tokens.")

            ## preparing human annotation form ##
            convert_data_to_tsv(data_dict, fout_csv)

    fin.close()
    fout.close()

    if args.output_csv:
        fout_csv.close()

def convert_data_to_tsv(data_dict, csv_writer):

    start_of_b = data_dict['labels'][1:].index(-1) + 1
    labelsB = data_dict['labels'][(start_of_b+1):-1]

    writting = []
    writting.append(f"{data_dict['idA']}#{data_dict['sentA']}")
    writting.append(f"{data_dict['idB']}#{data_dict['sentB']}")
    writting.append(f"TokensB#{'#'.join(data_dict['wordsB'])}")
    writting.append(f"Annotation-1#{'#'.join([str(l) for l in labelsB])}")
    writting.append(f"Annotation-2#{'#'.join([str(0) for l in labelsB])}")
    writting.append(f"Annotation-3#{'#'.join([str(0) for l in labelsB])}")
    writting.append(f"Annotation-4#{'#'.join([str(0) for l in labelsB])}")
    writting.append(f"Annotation-5#{'#'.join([str(0) for l in labelsB])}")
    csv_writer.write("\n".join(writting)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("--output_csv", default=False, action='store_true')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)
    convert_raw_to_highlight(args)

    print("Done")
