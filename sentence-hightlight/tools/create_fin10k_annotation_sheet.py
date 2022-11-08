# depreciated
import re
import os
import json
import argparse

def convert_data_to_tsv(args):

    fin = open(args.path_input_file, 'r')
    fout = open(args.path_output_file, 'w')

    with open(args.path_input_file, 'r') as fin:
        for i, line in enumerate(fin): 
            data_dict = json.loads(line.strip())

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
            fout.write("\n".join(writting)+'\n')

    fin.close()
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # For all data
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    args = parser.parse_args()

    convert_data_to_tsv(args)
    print("Done")
