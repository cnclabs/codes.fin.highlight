import argparse
from scipy import stats
from utils import load_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-truth", "--path_truth_file", type=str, default=None)
    parser.add_argument("-pred", "--path_pred_file", type=str)
    args = parser.parse_args()

    anchor = load_pred(arg.truth)
    references = load_pred(args.reference)

    with open(args.path_output_file. 'w') as f:
        pass

