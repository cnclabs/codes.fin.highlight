import collections
import argparse
import numpy as np
import json

# truth jsonl file
def load_truth_from_json(file_path, sentA=True):
    truth = collections.OrderedDict()
    sent = collections.OrderedDict()

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            try:
                pair_id = data['idA'] + "#" + data['idB']
            except:
                pair_id = i

            if sentA:
                truth[pair_id] = data['keywordsA']
            truth[pair_id] += data['keywordsB']
            sent[pair_id] = f"# {data['sentA']}\n# {data['sentB']}"

    return truth, sent

def load_pred_from_json(file_path, topk=None, prob_threshold=0, sentA=False):
    pred = collections.OrderedDict()
    prob = collections.OrderedDict()
    # punc = (lambda x: x in [",", ".", "?", "!"])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)

            try:
                pair_id = data['idA'] + "#" + data['idB']
            except:
                pair_id = i

            pred[pair_id] = []
            prob[pair_id] = []
            flag = None

            # consider the final threshold by topk and defined-threshold
            # [CONCERN] since the label @ sentence is not considered, cannot chose topk 
            # threshold = max(sorted(data['prob'], reverse=True)[:topk][-1], prob_threshold) 

            for j, (w, p) in enumerate(zip(data['words'], data['prob'])):
                if p == -1:
                    flag = sentA if j == 0 else True
                if (flag) and (p >= prob_threshold):
                    pred[pair_id].append(w)
                    prob[pair_id].append(p)

    return pred, prob
