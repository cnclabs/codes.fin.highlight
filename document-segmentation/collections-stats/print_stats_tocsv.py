from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer, BertModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--collection')
parser.add_argument('--output')

args = parser.parse_args()

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

# def get_num_of_seg(df):
#     df1 = df['fullID'].str.split('_', expand=True)
#     last_pid_indexes = [idx-1 for idx in df1[(df1[3]=='P0') & (df1[4]=='S0')].index][1:]
#     li = df1.loc[last_pid_indexes, 3].str.replace('P', '').astype('int')
#     last_para = int(df1.iloc[-1][3].replace('P', '')) # last
#     # for idx in last_pid_indexes[:20]:
#     #     print(idx, df1.loc[idx, 3])
#     return (sum(li)+last_para) / 800

def get_num_of_total_doc_and_company(data):
    fullIDs = list(data.keys())
    CIKs = []
    all_docs = []
    for fullID in fullIDs:
        cik, year, item, para, sent = fullID.split('_')
        all_docs.append(f'{cik}_{year}_{item}')
        CIKs.append(cik)
    CIKs = list(set(CIKs))
    all_docs = list(set(all_docs))
    return len(CIKs), len(all_docs)


def len_sent(tokenizer, text):
    return len(tokenizer.tokenize(text))

def get_sentence_length(df, tokenizer):
    df1 = df.copy()
    lengths = []
    for sent in tqdm(list(df1['sentence'])):
        lengths.append(len_sent(tokenizer, sent))
    df1['sent_length'] = lengths
    return df1

def get_segment_length(df):
    # create column: positioin
    # input already calculated sentence length df
    df1 = df.copy()
    df1['position'] = ['']*len(df)

    cur_pid = 0
    pos = 0
    positions = []
    for fullID in list(df['fullID']):
        cik, year, item, pid, sid = fullID.split('_')
        pid = int(pid.replace('P', ''))
        if cur_pid!=pid:
            pos+=1
            cur_pid = pid
        positions.append(pos)
    df1['position'] = positions
    
    df1['segment_length'] = [0]*len(df) # reset 0
    for pos in tqdm(positions):
        df3 = df1[df1['position']==pos]
        seg_length = sum(df3['sent_length'])
        df1.loc[df3.index, 'segment_length'] = seg_length
        # seg = " ".join([data[fullID] for fullID in df3.fullID])
        # df1.loc[df3.index, 'segment_length'] = len_sent(tokenizer, seg)
    
    return df1, positions

def main(args):
    # file = '/tmp2/cwlin/fintext-new/collections/8y-item7/rand100/8y_item7_further_preprocess_collections.txt'
    # pred_file = '/tmp2/cwlin/fintext-new/collections/prediction/rand100/8y_item7_further_preprocess_collections-100000'
    file = args.collection
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data, df = read_collections(file)
    total_company, total_docs = get_num_of_total_doc_and_company(data)
    print(f'total company: {total_company}')
    print(f'total docs: {total_docs}')
    df_sent = get_sentence_length(df, tokenizer)
    df_seg, positions = get_segment_length(df_sent)

    df_seg[['sent_length', 'position', 'segment_length']].to_csv(f'{args.output}', index=False)

if __name__=='__main__':
    main(args)