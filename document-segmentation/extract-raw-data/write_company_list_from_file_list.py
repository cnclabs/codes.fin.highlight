import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cmp_file_list', help='e.g., /tmp2/cwlin/fintext-new/codes.fin.contextualized/document-segmentation/extract-raw-data/8y-10k-company-file-list.txt')
parser.add_argument('--output', help='e.g., /tmp2/cwlin/fintext-new/codes.fin.contextualized/document-segmentation/extract-raw-data/8y-10k-company-list.txt')
args = parser.parse_args()

def main(args):
    cmp_list = []
    f = args.cmp_file_list
    output = args.output
    with open(f, 'r') as f:
        for line in f.readlines():
            cmp_list.append(line.split('_')[4])
    cmp_list = list(set(cmp_list))
    print("number of 8y 10k companies: ", len(cmp_list))

    f.close()

    with open(output, 'w') as f:
        for cmp in cmp_list:
            f.write(f'{cmp}\n')
    f.close()
    print('Done writing')

if __name__ == '__main__':
    main(args)