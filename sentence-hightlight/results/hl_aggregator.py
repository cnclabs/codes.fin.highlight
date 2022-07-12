"""
Function for aggregate highlight probabilities globalized
"""
import collections
import argparse
import numpy as np
import json
from utils import load_pred_from_json, load_truth_from_json

def average(x):
    return np.mean(x, axis=0).tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", "--path_pred_file", type=str, default='fin10k/type2.sentence.eval.results-cross-domain-transfer-0.125-12500')
    parser.add_argument("-out", "--path_output_file", type=str, default='aggregate.jsonl')
    parser.add_argument("-topk", "--topk", type=int, default=None)
    parser.add_argument("-hl_on_a", "--highlight_on_a", action='store_true', default=False)
    parser.add_argument("-thres", "--threshold", type=float, default=0)
    args = parser.parse_args()

    pred, prob = load_pred_from_json(args.path_pred_file,
                                     prob_threshold=args.threshold,
                                     sentA=args.highlight_on_a)

    prob_agg = collections.defaultdict(list)
    pair_agg = collections.defaultdict(list)
    feature_agg = collections.defaultdict(list)
    # id: <idA>#<idB>
    # collecting pairs

    for pair_id in pred:
        idA, idB = pair_id.split('#')
        features = pred[pair_id]
        importance = prob[pair_id]

        threshold = sorted(importance, reverse=True)[:args.topk][-1]

        if args.threshold is not None:
            threshold = max(args.threshold, threshold)

        importance = [i if i >= threshold else 0 for i in importance]

        pair_agg[idB].append(idA)
        feature_agg[idB] = features
        prob_agg[idB].append(importance)

    with open(args.path_output_file, 'w') as f:
        for idB in pair_agg:
            if len(pair_agg[idB]) > 1:
                print(pair_agg[idB], idB)
            f.write(json.dumps({
                "idB": idB,
                "rel_idAs": pair_agg[idB],
                "feature": feature_agg[idB],
                "importance": average(prob_agg[idB])
            })+'\n')

