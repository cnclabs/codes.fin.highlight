import collections
import argparse
import numpy as np
import json
from utils import load_pred, load_truth

def main(args):

    truth, text = load_truth(args.path_truth_file)
    prediction = load_pred(args.path_pred_file, args.threshold,)

    if len(truth) != len(prediction):
        print(f"[WARNING] Inconsisent sizes of truth and prediction, ",
                "{len(truth)} and {len(prediction)}")
        n = len([pk for pk in prediction.keys() if pk in truth.keys()])
        print(f"n examples annotated.")

    metrics = collections.defaultdict(list)

    # for i, truth_tokens in enumerate(truth.values()):
    i = 0
    for (pair_id, truth_tokens) in truth.items():

        # get topk
        pred_tokens = [t for (t, p) in sorted(prediction[pair_id], key=lambda x: x[1], reverse=True)][:args.topk]

        n_truth = len(truth_tokens)
        n_pred = len(pred_tokens)

        hits = set(truth_tokens) & set(pred_tokens)
        hits_recall = set(pred_tokens[:n_truth]) & set(truth_tokens)

        precision = (len(hits) / n_pred) if n_pred != 0 else 0
        recall = (len(hits) / n_truth) if n_truth != 0 else 0
        r_precision = len(hits_recall) / n_truth if n_truth != 0 else 0

        if precision + recall != 0:
            fscore = 2 * precision * recall / (precision + recall)
        else:
            fscore = 0

        if len(truth_tokens) != 0:
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(fscore)
            metrics['rp'].append(r_precision)
            i += 1

            if args.verbose:
                print(f"{text[pair_id]}\
                        \n - Performance: (R: {recall}; P: {precision})\
                        \n - Truth: {truth_tokens}\
                        \n - Predict: {pred_tokens}")

    print("********************************\
            \nFile: {}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nNum of evaluated samples: {}\
            \n********************************".format( 
                args.path_pred_file.split('/')[1],
                'precision', np.mean(metrics['precision']), 
                'recall', np.mean(metrics['recall']), 
                'f1-score', np.mean(metrics['f1']),
                'Rprecision', np.mean(metrics['rp']), i+1
            ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default=None)
    parser.add_argument("-pred", "--path_pred_file", type=str)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    parser.add_argument("-topk", "--topk", type=int, default=None)
    args = parser.parse_args()

    main(args)
