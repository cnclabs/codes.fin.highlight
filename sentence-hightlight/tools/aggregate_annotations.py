import collections
import argparse
import numpy as np
import json
from utils import load_json

for type in ['type2', 'type1.easy', 'type1.hard']:

    # three annotators
    annotations = {
            "annotator1": load_json(f"data/fin10k/fin10k.annotation.{type}.jsonl.1"),
            "annotator2": load_json(f"data/fin10k/fin10k.annotation.{type}.jsonl.2"),
            "annotator3": load_json(f"data/fin10k/fin10k.annotation.{type}.jsonl.3"),
    }

    annotated_pair_ids = list(annotations['annotator1'].keys())

    # sanity check
    if len([i for i in annotated_pair_ids if i not in annotations['annotator2'].keys()]) != 0:
        print('Inconsistent')
    if len([i for i in annotated_pair_ids if i not in annotations['annotator3'].keys()]) != 0:
        print('Inconsistent')

    # touch a file
    f = open(f"data/fin10k/fin10k.annotation.{type}.jsonl", 'w')

    for pair_id in annotated_pair_ids:

        counter_keywords = collections.Counter()
        # counter_WP = collections.Counter()
        W, P = map(list, list(zip(*annotations[f"annotator1"][pair_id]['WP'])))
        aggregation = annotations[f"annotator1"][pair_id]

        for annotator_i in [2,3]:
            # keywords adding
            counter_keywords += collections.Counter(
                    annotations[f"annotator{annotator_i}"][pair_id].pop('keywords')
            )

            # aggrement adding
            W_, P_ = map(list, list(zip(*annotations[f"annotator{annotator_i}"][pair_id]['WP'])))
            assert len(W) == len(W_), 'Inconsistent lenght of words'
            P = P + np.array(P_)

        # keywords aggregated:w
        aggregation.update({
            "idB": pair_id.split("#")[1], "idA": pair_id.split("#")[0],
            "keywords": [k for k, v in counter_keywords.items() if v >= 2],
            "WP": (P/3).tolist()
        })
        f.write(json.dumps(aggregation) + '\n')

    f.close()

