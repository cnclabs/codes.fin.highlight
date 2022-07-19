import collections
import argparse
import numpy as np
import json
from utils import load_truth_from_json, load_pred_from_json

def main(args):

    truth, strings = load_truth_from_json(args.path_truth_file)
    pred, prob = load_pred_from_json(
            args.path_pred_file,
            prob_threshold=args.threshold,
            sentA=args.highlight_on_a
    )
    path_output_file = args.path_output_file + f".{args.path_pred_file.split('results-')[1]}"
    fout = open(path_output_file, 'w')

    for pair_id, truth_tokens in truth.items():

        topk = list(zip(pred[pair_id], prob[pair_id]))
        topk = sorted(topk, key=lambda x: x[1], reverse=True)
        topk = topk[:args.topk]

        fout.write(f'{strings[pair_id]}\n')
        string_topk = ", ".join([f"{i}: {round(j, 3)}" for (i, j) in topk])
        fout.write(f"TOP K: {string_topk}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default='')
    parser.add_argument("-pred", "--path_pred_file", type=str, default='')
    parser.add_argument("-out", "--path_output_file", type=str, default='')
    parser.add_argument("-hl_on_a", "--highlight_on_a", action='store_true', default=False)
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    parser.add_argument("-topk", "--topk", type=int, default=None)
    args = parser.parse_args()

    main(args)
