import collections
import argparse
import numpy as np
import json
from utils import load_json, aggregate_raters, fleiss_kappa

def main(args):

    # loading
    annotation_list = []
    for i in args.aggregate:
        if len(args.files) == 0:
            annotation = load_json(args.path_truth_file+f".{i}")
        else:
            annotation = {}
           for file in args.files:
                annotation.update(load_json(f"{file}.{i}"))

        annotation_list.append(annotation)

    # print([p for (w, p) in annotation_list[0]['845877_13_item7_p93_s0#845877_14_item7_p75_s3']['WP']])

    # flatten
    annotation_labels = collections.defaultdict(list)
    for pair_id in annotation_list[0]:
        for i in range(len(args.aggregate)):
            annotation_labels[i] += [p for (w, p) in annotation_list[i][pair_id]['WP']]

    # annotation
    table = np.concatenate([[annotation_labels[i]] for i in range(len(args.aggregate))])

    # aggregator
    table = table.transpose()
    print(table.shape)
    return fleiss_kappa(aggregate_raters(table)[0], method='fleiss')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default=None)
    parser.add_argument("--files", default=[], action='append')
    parser.add_argument("--aggregate", default=[], action='append')
    args = parser.parse_args()

    kappa = main(args)
    print(kappa)
