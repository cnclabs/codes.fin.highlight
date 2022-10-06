import collections
import argparse
import numpy as np
import json
from utils import load_pred, load_truth, load_json, aggregate_annotation

def highlight_eval(args):

    if len(args.aggregate) > 1:
        truth_list = []
        for i in args.aggregate:
            truth_list.append(load_json(args.path_truth_file+f".{i}"))
        truth = aggregate_annotation(truth_list)
        with open(args.path_truth_file, 'w') as f:
            for pair_id in truth:
                f.write(json.dumps(truth[pair_id]) + '\n')
    else:
        truth = load_json(args.path_truth_file)
    pred = load_json(args.path_pred_file)

    if len(truth) != len(pred):
        print(f"[WARNING] Inconsisent sizes of truth and prediction, ",
                "{len(truth)} and {len(pred)}")
        n = len([pk for pk in pred.keys() if pk in truth.keys()])
        print(f"n examples annotated.")

    metrics = collections.defaultdict(list)

    # for i, truth_tokens in enumerate(truth.values()):
    i = 0
    for (pair_id, truth_dict) in truth.items():

        # get topk
        truth_tokens = truth_dict['keywords']
        pred_tokens = [w for (w, p) in sorted(pred[pair_id]['WP'], key=lambda x: x[1], reverse=True)][:args.topk]
        pred_probs = [p for (w, p) in pred[pair_id]['WP']]
        truth_probs = [p for (w, p) in truth_dict['WP']]

        text_pair = truth_dict['text_pair']

        n_truth = len(truth_tokens)
        n_pred = len(pred_tokens)

        hits = set(truth_tokens) & set(pred_tokens)
        hits_recall = set(truth_tokens) & set(pred_tokens[:n_truth]) 

        precision = (len(hits) / n_pred) if n_pred != 0 else 0
        recall = (len(hits) / n_truth) if n_truth != 0 else 0
        r_precision = len(hits_recall) / n_truth if n_truth != 0 else 0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall != 0) else 0

        if len(truth_dict) != 0:
            metrics['precision'].append(precision)
            if n_truth != 0:
                i += 1
                metrics['rp'].append(r_precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(fscore)

                if np.std(truth_probs) != 0:
                    correlation = np.corrcoef(truth_probs, pred_probs)[0, 1] if n_truth != 0 else 0
                    metrics['pearson'].append(correlation)
                    # less 1 examples

            if args.verbose:
                print(f"{text_pair}\
                        \n - Performance\
                        \n - Recall: {recall}; P: {precision}\
                        \n - R-Prec: {r_precision}\
                        \n - Pearson: {correlation}\
                        \n - Truth: {truth_tokens}\
                        \n - Predict: {pred_tokens}")

    print("********************************\
            \nFile: {}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nNum of evaluated samples: {}\
            \n********************************".format( 
                args.path_pred_file.split('/')[-1],
                'precision', np.nanmean(metrics['precision']), 
                'recall', np.nanmean(metrics['recall']), 
                'f1-score', np.nanmean(metrics['f1']),
                'Pearson', np.nanmean(metrics['pearson']),
                'RPrecision', np.nanmean(metrics['rp']), i+1
            ))

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default=None)
    parser.add_argument("-pred", "--path_pred_file", type=str)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--aggregate", default=[], action='append')
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    parser.add_argument("-topk", "--topk", type=int, default=None)
    args = parser.parse_args()

    highlight_eval(args)
