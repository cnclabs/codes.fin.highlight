import argparse
import pandas as pd
import os
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--collection_file')
parser.add_argument('--output_dir')
args = parser.parse_args()

# file = '/tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-collections.txt'
# output = '/tmp2/cwlin/fintext-new/collections/8y-item7/company'

def read_collections(file):
    data = dict()
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fullID, sentence = line.split('\t')
            data[fullID] = sentence
        f.close()

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df.columns = ['fullID', 'sentence']
    return data, df

def main(args):
    
    file = args.collection_file
    output = args.output_dir
    data,df = read_collections(file)

    df1 = df['fullID'].str.split('_', expand=True)
    df1.columns = ['cik', 'year', 'item', 'para', 'sent']

    for cik in tqdm(list(set(df1['cik']))):
        df_cmp_collection = df.loc[df1[df1['cik']==cik].index]
        os.makedirs(f'{output}/{cik}', exist_ok=True)
        df_cmp_collection.to_csv(f'{output}/{cik}/collections.txt', sep='\t', header=None, index=False, quoting=csv.QUOTE_NONE)

if __name__ == "__main__":
    main(args)