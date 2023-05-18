import collections
import argparse
import numpy as np
import json
from utils import load_json, aggregate_annotation

def load_data(path):

    out_dict = {}
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            data_dict = {}
            data = json.loads(line)
            pair_id = (data.pop('idA'), data.pop('idB'))
            # data_dict.update({'pair_id': pair_id})
            data_dict.update(data)
            out_dict[pair_id] = data_dict

    return out_dict

def get_aggregated_labels(args):

    truth_list = []
    for i in [1,2,3]:
        truth_list.append(load_data(args.path_truth_file+f".{i}"))

    fout = open(args.path_truth_file+'.truth', 'w')

    for pair_id in truth_list[0]:
        data = np.array(truth_list[0][pair_id]['probs'])
        data = data + truth_list[1][pair_id]['probs']
        data = data + truth_list[2][pair_id]['probs']

        # labels
        labels = [-1 if l < 0 else int(l) for l in data]
        print(labels)

        # probs
        probs = [p for p in data/3]

        truth_list[0][pair_id].update(
                {"probs": probs, "labels": labels}
        )

        fout.write(json.dumps(truth_list[0][pair_id])+'\n')

        print(truth_list[0][pair_id])

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default=None)
    # parser.add_argument("-pred", "--path_pred_file", type=str)
    # parser.add_argument("--aggregate", default=[], action='append')
    args = parser.parse_args()

    get_aggregated_labels(args)
