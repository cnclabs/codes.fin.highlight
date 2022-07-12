import collections
import argparse
import numpy as np
import json
from utils import load_pred_from_json, load_truth_from_json

def main(args):
    if args.validate_split is not None:
        path_truth_file = args.path_truth_file.replace('split', args.validate_split)
    else:
        path_truth_file = args.path_truth_file

    truth, strings = load_truth_from_json(path_truth_file)
    pred, prob = load_pred_from_json(
            args.path_pred_file,
            prob_threshold=args.threshold,
            sentA=args.highlight_on_a
    )

    if len(truth) != len(pred):
        print(f"[WARNING] Inconsisent sizes of truth and predictions, {len(truth)} and {len(pred)}")
        n = len([pk for pk in pred.keys() if pk in truth.keys()])
        print(f"n examples annotated.")

    metrics = collections.defaultdict(list)

    # for i, truth_tokens in enumerate(truth.values()):
    i = 0
    for (pair_id, truth_tokens) in truth.items():

        try:
            pred_tokens = pred[pair_id]
            prob_tokens = prob[pair_id]
        except:
            pred_tokens = []

        # get topk
        pred_tokens = [t for (t, p) in sorted(zip(pred_tokens, prob_tokens), key=lambda x: x[1], reverse=True)[:args.topk]]

        hits = set(truth_tokens) & set(pred_tokens)
        precision = (len(hits) / len(pred_tokens)) if len(pred_tokens) != 0 else 0
        recall = (len(hits) / len(truth_tokens)) if len(truth_tokens) != 0 else 0

        if precision + recall != 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0

        if len(truth_tokens) != 0:
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(fscore)
            i += 1

            if args.verbose:
                print(f"{strings[pair_id]}\
                        \n - Performance: (R: {recall}; P: {precision})\
                        \n - Truth: {truth_tokens}\
                        \n - Predict: {pred_tokens}")

    print("********************************\
            \nFile: {}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nNum of evaluated samples: {}\
            \n********************************".format( 
                args.path_pred_file.split('/')[1],
                'precision', np.mean(metrics['precision']), 
                'recall', np.mean(metrics['recall']), 
                'f1-score', np.mean(metrics['f1']), i+1
            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-split", "--validate_split", type=str, default=None)
    parser.add_argument("-truth", "--path_truth_file", type=str, \
            default='../data/esnli/esnli.split.sent_highlight.contradiction.jsonl')
    parser.add_argument("-pred", "--path_pred_file", type=str)
    parser.add_argument("-hl_on_a", "--highlight_on_a", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    parser.add_argument("-k", "--topk", type=int, default=None)
    args = parser.parse_args()

    main(args)
