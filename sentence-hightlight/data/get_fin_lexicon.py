import argparse
import pandas as pd
from scripts.utils import load_stopwords

def main(args):

    data = pd.read_csv(args.path_lm_master_dict)
    data = data[(data['Doc Count'] > 0) & (data['Word Count'] > 0)]
    data.dropna(inplace=True)
    dropped_columns = ['Seq_num', 'Word Proportion', 'Average Proportion', 'Std Dev']
    dropped_columns += ['Syllables', 'Source']
    data = data.drop(columns=dropped_columns)
    data['Word'] = data['Word'].apply(lambda x: x.casefold())

    # file out dictionary
    f_sentiment = open('LM.master_dictionary.sentiment.dict', 'w')
    f_stopword = open('LM.master_dictionary.stopwords.dict', 'w')

    # sentiment dict
    data_sentiment = data[(data.iloc[:, -7:]!= 0).sum(1).astype(bool)]
    data_non_sentiment = data[~((data.iloc[:, -7:]!= 0).sum(1).astype(bool))]

    # output the sentiment wordlist
    data_sentiment.sort_values(
            by=['Doc Count', 'Word Count', 'Word'], ascending=False, ignore_index=True
    )
    for ind, row in data_sentiment.iterrows():
        f_sentiment.write("{}\t{}\t{}\n".format(
            row['Word'], row['Word Count'], row['Doc Count']
        ))
    print(data_sentiment)

    # output the financial stopwords
    stopwords = load_stopwords('anserini')

    # get the reciprocal rank of "word count" and "doc count", caculate rank fusion
    data_non_sentiment = data_non_sentiment[['Word', 'Word Count', 'Doc Count']]
    for col in ['Word Count', 'Doc Count']:
        data_non_sentiment[col] = data_non_sentiment[col].rank(ascending=False).astype(int)

    data_non_sentiment['RR'] = \
            (1/data_non_sentiment['Word Count'] + 1/data_non_sentiment['Doc Count']).round(4)
    data_non_sentiment.sort_values(by=['RR'], ascending=False, inplace=True, ignore_index=True)

    threshold = data_non_sentiment.index.where(data_non_sentiment['Word'].isin(stopwords)).max() 
    # the lowest rank of the stopwords appeared in the financail dictionaryc:W

    for ind, row in data_non_sentiment.iterrows():
        if row['Word'] in stopwords:
            pass
        else:
            f_stopword.write("{}\t{}\t{}\t{}\n".format(
                row['Word'], row['RR'], row['Word Count'], row['Doc Count']
            ))
        if isinstance(args.truncated_by_k, int) and (ind == args.truncated_by_k):
            break
        elif (args.truncated_by_k is None) and (ind >= threshold):
            break

    print(data_non_sentiment[data_non_sentiment['Word'].isin(stopwords)])
        
parser = argparse.ArgumentParser()
parser.add_argument("-LM_master_dict", "--path_lm_master_dict", type=str, default='Loughran-McDonald_MasterDictionary_1993-2021.csv')
parser.add_argument("-topk", "--truncated_by_k", type=str, default=None)
args = parser.parse_args()

main(args)
print('done')
