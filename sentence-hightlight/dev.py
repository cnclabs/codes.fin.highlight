import collections
import argparse
from scipy import stats
from tools.utils import load_pred, load_json
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, 
            default='data/fin10k/fin10k.annotation.type2.jsonl.1')
    parser.add_argument("-pred1", "--path_ref1_file", type=str,
            default='results/fin10k.eval/type2/fin10k.eval.type2.results-esnli-zs-highlighter')
    parser.add_argument("-pred2", "--path_ref2_file", type=str,
            default='results/fin10k.eval/type2/fin10k.eval.type2.results-further-finetune-sl-smooth')
    parser.add_argument("-output", "--path_output_file", type=str, default='rm.me')
    args = parser.parse_args()

    anchor = {k: [p for (w, p) in v['WP']] for k, v in load_json(args.path_truth_file).items()}
    ref1 = {k: [p for (w, p) in v['WP']] for k, v in load_json(args.path_ref1_file).items()}
    ref2 = {k: [p for (w, p) in v['WP']] for k, v in load_json(args.path_ref2_file).items()}

    scores = {'ref1':[], 'ref2':[]}
    with open(args.path_output_file, 'w') as f:
        i = 0
        for pairid in anchor:
            # pearson r
            # scores.append(
            #         stats.pearsonr(anchor[pairid], ref[pairid])
            # )
            scores_ref1 = np.corrcoef(anchor[pairid], ref1[pairid])[1, 0]
            scores_ref2 = np.corrcoef(anchor[pairid], ref2[pairid])[1, 0]

            if np.isnan(scores_ref1) or np.isnan(scores_ref2):
                pass
            else:
                i += 1
                scores['ref1'].append(scores_ref1)
                scores['ref2'].append(scores_ref2)
                # print(pairid, ":",  anchor[pairid])

    # print(np.isnan(scores['ref1']).sum())
    # print(np.isnan(scores['ref2']).sum())

    # see the detail or (t-test)
    print(np.array(scores['ref1']))
    print(np.array(scores['ref2']))

    print(i+1)
    print(np.nanmean(scores['ref1']))
    print(np.nanmean(scores['ref2']))

