import argparse
from utils import load_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", "--path_pred_file", type=str, default=None)
    parser.add_argument("-topk", "--topk", type=int, default=5-1)
    args = parser.parse_args()

    data = load_json(args.path_pred_file)
    for pairs in data:
        print(data[pairs]['text_pair'])
        topk = min(args.topk, len(data[pairs]['WP'])-1)
        threshold = sorted(data[pairs]['WP'], key=lambda x: x[1], reverse=True)[topk][1]
        print("Top-K highlighted:")
        words = [f"*{w}*" if p >= threshold else f"{w}" for (w, p) in data[pairs]['WP']]
        print(" ".join(words))

