from more_itertools import chunked
import re
import os
import json
import argparse
import collections

def convert_raw_to_highlight(args):

    # fraw = open(args.path_raw_file, 'r')
    # fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')
    annotations = collections.defaultdict(dict)

    # laod input annotation
    with open(args.path_input_file, 'r') as fin:
        for line in chunked(fin, 8):
            # pair id
            pairs_id = f"{line[1].split(',', 1)[0]}-{line[0].split(',', 1)[0]}"
            
            # annotaion
            for line_annotation in line[3:]:
                annotator_i, annotation = line_annotation.strip().split(',', 1)
                i = annotator_i.split('-')[1]

                if i in args.annotators:
                    annotations[pairs_id][i] = [int(l) for l in annotation.split(',') if l != ""]

    # annotataion comprehensiveness
    # laod raw
    highlight = 0
    with open(args.path_raw_file, 'r') as fraw:
        for n, line in enumerate(fraw): 

            data_dict = json.loads(line.strip())
            idA, idB = data_dict['idA'], data_dict['idB']
            wordsA, wordsB = data_dict['wordsA'], data_dict['wordsB']

            if len(args.annotators) == 1:
                i = args.annotators[0]
                labelsB = annotations[f"{idB}-{idA}"][i]
            else:
                # [TODO] Aggregate the annotations
                for i in args.annotators:
                    labelsB_i = annotations[f"{idB}-{idA}"][i]

            # labels
            labels = [-1] + [0]*len(wordsA) + [-1] + labelsB + [-1]
            probs = [-1] + [0]*len(wordsA) + [-1] + labelsB + [-1]

            # keywords
            assert len(wordsB) == len(labelsB), 'Inconsistent lenght of words'
            tokensB_hl = [w for (w, l) in zip(wordsB, labelsB) if l == 1]

            data_dict.update({
                'keywordsA': [], 'keywordsB': tokensB_hl,
                'labels': labels, 'probs': probs
            })
            highlight += 0
            highlight += len(tokensB_hl) 

            fout.write(json.dumps(data_dict) + '\n')
            print(f"{n+1} annotation examples with {highlight} highlighted tokens.")

    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-raw", "--path_raw_file", type=str)
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("-annotator", "--annotators", action='append')
    args = parser.parse_args()

    # makedirs
    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)

    convert_raw_to_highlight(args)

    print("Done")
