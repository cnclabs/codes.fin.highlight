import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--sent_and_seg_length_file', help='csv file')
args = parser.parse_args()

def print_stats(df):
    # number of sentences per documents
    print('number of total documents', 800)
    print('number of total sentences', len(df))
    print('number of total segment/paragraphs', df['position'].iloc[-1]+1)
    print('number of sentences per document', len(df) / 800)
    print('number of segment/paragraph per document', (df['position'].iloc[-1]+1) / 800)

    # s = 0
    # for i in list(set(df.position)):
    #     s+=df[df['position']==i]['segment_length'].values[0]
    # print('total number of sentence tokens', df['sent_length'].sum())
    # print('total number of paragraph tokens', s)
    print('average number of tokens per sentence', df['sent_length'].mean())
    print('average number of tokens per segment/paragraph', df['sent_length'].sum() / (df['position'].iloc[-1]+1))

    print('Number of long sentence (>256 tokens): ', len(df[df['sent_length']>256]))
    print('Number of long sentence (>512 tokens): ', len(df[df['sent_length']>512]))

    count_long_seg = 0
    count_super_long_seg = 0
    for i in list(set(df.position)):
        seg_length = df[df['position']==i]['segment_length'].values[0]
        if seg_length>256:
            count_long_seg+=1
        if seg_length>512:
            count_super_long_seg+=1
    print('Number of long segment/paragraph (>256 tokens): ', count_long_seg)
    print('Number of long segment/paragraph (>512 tokens): ', count_super_long_seg)

def main(args):
    f = args.sent_and_seg_length_file
    df = pd.read_csv(f)
    print_stats(df)

if __name__ == '__main__':
    main(args)