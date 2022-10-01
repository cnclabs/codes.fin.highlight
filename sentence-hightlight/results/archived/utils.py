import collections
import numpy as np
import json

def load_truth(file_path, sentA=True):
    truth = {}

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            try:
                pair_id = data['idA'] + "#" + data['idB']
            except:
                pair_id = i

            truth[pair_id] = {
                    "keywords": data['keywordsB'], 
                    "text_pair": f"# {data['sentA']}\n# {data['sentB']}"
            }

    return truth

def load_pred(file_path, special_token=False, prob_threshold=0):
    prediction = {}
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            try:
                pair_id = data.pop('idA') + "#" + data.pop('idB')
            except:
                pair_id = i

            prediction[pair_id] = []

            flag = False
            for j, (w, p) in enumerate(zip(data['words'], data['probs'])):

                if p == -1:
                    # when aggregation
                    if special_token:
                        prediction[pair_id].append( (w, p) )
                        flag = True
                    # when evalaution
                    else:
                        flag = False if j == 0 else True
                elif flag:
                    if p >= prob_threshold:
                        prediction[pair_id].append( (w, p) )
                    else:
                        prediction[pair_id].append( (w, 0) )
                else:
                    pass

    return prediction
            # consider the final threshold by topk and defined-threshold
            # [CONCERN] since the label @ sentence is not considered, cannot chose topk 
            # threshold = max(sorted(data['prob'], reverse=True)[:topk][-1], prob_threshold) 
