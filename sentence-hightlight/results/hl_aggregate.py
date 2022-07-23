"""
Function for aggregate highlight probabilities globalized
"""
import os
import collections
import argparse
import numpy as np
import json
from utils import load_pred_from_json

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
                    'probs': np.array(importances)
            }
        else:
            # Noted that the words in idA-2,3,4.... will never be appended
            predictions_agg[idB]['idA'].append(idA)
            anchor = np.argwhere(predictions_agg[idB]['probs']==-1).flatten()[1] # start of sentence B
            anchor_other = np.argwhere(np.array(importances)==-1).flatten()[1] # start of sentence B
            predictions_agg[idB]['probs'][(anchor+1):] += importances[(anchor_other+1):]

    with open(args.path_output_file, 'w') as f:
        for idB in predictions_agg:
            n_pairs = len(predictions_agg[idB]['idA'])
            if n_pairs > 1:
                print(predictions_agg[idB]['idA'], idB)
                # post-process (average with first dimension)
                anchor = np.argwhere(predictions_agg[idB]['probs']==-1).flatten()[1] # start of sentence B
                predictions_agg[idB]['probs'][(anchor+1):] = predictions_agg[idB]['probs'][(anchor+1):] / n_pairs

            predictions_agg[idB]['probs'] = predictions_agg[idB]['probs'].tolist()

            f.write(json.dumps(predictions_agg[idB])+'\n')

