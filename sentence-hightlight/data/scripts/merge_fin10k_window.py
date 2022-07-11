import parser
import collections



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--path_input_file", type=str)
    parser.add_argument("-output", "--path_output_file", type=str)
    parser.add_argument("-window", "--path_output_file", type=str)
    args = parser.parse_args()
