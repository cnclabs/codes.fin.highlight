import pandas as pd
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--collection') # '/tmp2/cwlin/fintext-new/collections/all/collections.txt'
parser.add_argument('--output') # '/tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-collections.txt'
parser.add_argument('--output_cmp') # '/tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-company-(more than 5 sentences).txt'
args = parser.parse_args()

def read_collections(file):
    data = dict()
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fullID, sentence = line.split('\t')
            data[fullID] = sentence
        f.close()
    return data

def main(args):
    file = args.collection
    output = args.output # 8y-item7-collections.txt
    output_cmp = args.output_cmp
    # print(file, output, output_cmp)
    data = read_collections(file)

    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df.columns = ['fullID', 'sentence']

    # # filter only ITEM7
    # df1 = df['fullID'].str.split('_', expand=True)
    # df1.columns = ['cik', 'year', 'item', 'para', 'sent']

    # item7_indexs = list(df1[df1['item']=='ITEM7'].index)
    # df_item7 = df.loc[item7_indexs]


    # # find companies that has item7 (with more than 5 sentences) throughout 8 years
    # df2 = df_item7['fullID'].str.split('_', expand=True)
    # df2.columns = ['cik', 'year', 'item', 'para', 'sent']

    # df2_count = df2.groupby(['cik', 'year', 'item']).count()
    # df2_count_sent = df2_count[df2_count['sent']>5]

    # cik_item7_8y_more_than_5_sent = []
    # for cik in list(set(df2_count_sent.index.get_level_values('cik'))):
    #     if len(df2_count_sent.loc[cik])==8: # that more than 5 sentences and must have 8 years
    #         cik_item7_8y_more_than_5_sent.append(cik)

    dfx = df['fullID'].str.split('_', expand=True)
    dfx.columns = ['cik', 'year', 'item', 'para', 'sent']
    print('Number of companies after parse: ', len(list(set(dfx.cik))))

    # must have 8 years, regardless what items there are
    dfx_count = dfx.groupby(['cik', 'year', 'item']).count()
    cik_8y = []
    for cik in list(set(dfx_count.index.get_level_values('cik'))):
        if len(set(dfx_count.loc[cik].index.get_level_values('year')))==8: 
            cik_8y.append(cik)
    print('Company that has 8 years of items (regardless what items): ', len(cik_8y))

    item7_indexs = list(dfx[dfx['item']=='ITEM7'].index)
    df_item7 = df.loc[item7_indexs]
    df2 = df_item7['fullID'].str.split('_', expand=True)
    df2.columns = ['cik', 'year', 'item', 'para', 'sent']

    df2_count = df2.groupby(['cik', 'year', 'item']).count()
    cik_item7_8y = []
    for cik in list(set(df2_count.index.get_level_values('cik'))):
        if len(df2_count.loc[cik])==8: # must have 8 years
            cik_item7_8y.append(cik)
    print('Company that must have 8 years of Item 7: ', len(cik_item7_8y))


    df2_count_sent = df2_count[df2_count['sent']>5]
    cik_item7_8y_more_than_5_sent = []
    for cik in list(set(df2_count_sent.index.get_level_values('cik'))):
        if len(df2_count_sent.loc[cik])==8: # that more than 5 sentences and must have 8 years
            cik_item7_8y_more_than_5_sent.append(cik)
    print('Company that has 8 years of Item 7, and Item 7 must have more than 5 sentences: ', len(cik_item7_8y_more_than_5_sent))


    with open(output_cmp, 'w') as f:
        for cik in cik_item7_8y_more_than_5_sent:
            f.write(f'{cik}\n')
    f.close()

    # find those index
    indexes = []
    for cik in cik_item7_8y_more_than_5_sent:
        indexes+=list(df2[df2['cik']==cik].index)

    df_item7_more_than_5_8y = df_item7.loc[indexes]
    df_item7_more_than_5_8y = df_item7_more_than_5_8y.sort_index()
    df_item7_more_than_5_8y.to_csv(output, sep='\t', header=None, index=False, quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main(args)