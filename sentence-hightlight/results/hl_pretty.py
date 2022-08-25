import collections
import argparse
import numpy as np
import json
from utils import load_truth, load_pred

def main(args):

    truth, text = load_truth(args.path_truth_file)
    prediction = load_pred(args.path_pred_file, args.threshold,)

    path_output_file = args.path_output_file + f".{args.path_pred_file.rsplit('/')[1]}"
    fout = open(path_output_file, 'w')

    for pair_id, truth_tokens in truth.items():

        pred = sorted(prediction[pair_id], key=lambda x: x[1], reverse=True)[:args.topk]

        fout.write(f'{text[pair_id]}\n')
        output_topk = ", ".join([f"({i}: {round(j, 10)})" for (i, j) in pred])
        fout.write(f">> {output_topk}\n")

    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default='')
    parser.add_argument("-pred", "--path_pred_file", type=str, default='')
    parser.add_argument("-out", "--path_output_file", type=str, default='')
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    parser.add_argument("-topk", "--topk", type=int, default=None)
    args = parser.parse_args()

    main(args)
