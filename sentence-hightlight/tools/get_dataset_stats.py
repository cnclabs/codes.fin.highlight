import collections
import json
import argparse
        
def main(args):
    fin = open(args.path_dataset, 'r')
    stats = collections.defaultdict(list)
    for n, jsonl in enumerate(fin):
        data_dict = json.loads(jsonl.strip())
        stats['sentA_length'].append(len(data_dict['wordsA']))
        stats['sentB_length'].append(len(data_dict['wordsB']))
        stats['pos_label'].append(sum([1 for l in data_dict['labels'] if l == 1]))
        stats['neg_label'].append(sum([1 for l in data_dict['labels'] if l == 0]))
        stats['no_label'].append(sum([1 for l in data_dict['labels'] if l == -100]))

    return stats, n+1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--path_dataset", type=str, \
            default='data/esnli/esnli.train.sent_highlight.contradiction.jsonl')
    args = parser.parse_args()
    results, n_examples = main(args)
    
    print(f'n_examples: {n_examples}')
    for k in results:
        print(f'{k}: {sum(results[k]) / n_examples}')
