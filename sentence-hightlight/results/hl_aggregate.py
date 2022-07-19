"""
Function for aggregate highlight probabilities globalized
"""
import os
import collections
import argparse
import numpy as np
import json
from utils import load_pred_from_json

def average(x, n):
    return np.array(x).reshape(-1, n).mean(0).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", "--path_pred_file", type=str, \
            default='fin10k/fin10k.eval.type2.segments.results-from-scratch-101')
    parser.add_argument("-out", "--path_output_file", type=str, default='aggregate.jsonl')
    parser.add_argument("-topk", "--topk", type=int, default=None)
    parser.add_argument("-hl_on_a", "--highlight_on_a", action='store_true', default=False)
    parser.add_argument("-thres", "--threshold", type=float, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.path_output_file), exist_ok=True)

    predictions = load_pred_from_json(
            args.path_pred_file,
            prob_threshold=args.threshold,
            sentA=args.highlight_on_a
    )
    predictions_agg = collections.defaultdict(dict)

    # prob_agg = collections.defaultdict(list)
    # pair_agg = collections.defaultdict(list)
    # words_agg = collections.defaultdict(list)
    # id: <idA>#<idB>
    # collecting pairs

    for pair_id in predictions:
        idA, idB = pair_id.split('#')

        importances = predictions[pair_id]['probs']

        threshold = sorted(importances, reverse=True)[:args.topk][-1]
        if args.threshold is not None:
            threshold = max(args.threshold, threshold)
        importances = [i if i >= threshold else 0 for i in importances]

        if idB not in predictions_agg:
            predictions_agg[idB] = {
                    'type': predictions[pair_id]['type'],
                    'idA': [idA],
                    'idB': idB,
                    'words': predictions[pair_id]['words'],
                    'probs': importances
            }
        else:
            predictions_agg[idB]['idA'].append(idA)
            predictions_agg[idB]['probs'] += importances

    with open(args.path_output_file, 'w') as f:
        for idB in predictions_agg:
            if len(predictions_agg[idB]['idA']) > 1:
                print(predictions_agg[idB]['idA'], idB)
                # post-process (average with first dimension)
                predictions_agg[idB]['probs'] = average(
                        predictions_agg[idB]['probs'], len(predictions_agg[idB]['idA'])
                )

            f.write(json.dumps(predictions_agg[idB])+'\n')

