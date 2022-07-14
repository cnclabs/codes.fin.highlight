import regex as re
import pandas as pd
from transformers import BertTokenizer, BertModel
import argparse
import csv
from tqdm import tqdm
import spacy

pd.options.mode.chained_assignment = None  # default='warn'
parser = argparse.ArgumentParser()
parser.add_argument('--collection') # /tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-collections.txt
parser.add_argument('--output') # /tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-collections_further_preprocess.txt
parser.add_argument('--sent_length') # /tmp2/cwlin/fintext-new/collections/8y-item7/8y-item7-collections_further_preprocess.txt

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

def len_sent(tokenizer, text):
    return len(tokenizer.tokenize(text))

def get_length(df, tokenizer):
    lengths = []
    for sent in tqdm(list(df['sentence'])):
        lengths.append(len_sent(tokenizer, sent))
    df['sent_length'] = lengths
    return df

def fur_preprocess(df):
    # 1 Begin with "Item 7 . "
    # fur_preprocess_ids_1 = list(df[df.sentence.str.contains('^(Item \d+ . )', regex=True)].index)
    df['sentence'] = df.sentence.str.replace('^(Item \d+ . )', repl='', regex=True)
    
    # 2 Begin with "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    # fur_preprocess_ids_2 = list(df[df.sentence.str.contains("^(MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS )", regex=True)].index)
    df['sentence'] = df.sentence.str.replace("^(MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS)", repl='', flags=re.IGNORECASE, regex=True)
    df['sentence'] = df.sentence.str.replace("^(MANAGEMENT S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS)", repl='', flags=re.IGNORECASE, regex=True)
    df['sentence'] = df.sentence.str.replace("^(EXECUTIVE SUMMARYq)", repl='', flags=re.IGNORECASE, regex=True)
    
    # 3 Begin with "Table of Contents"
    # fur_preprocess_ids_3 = list(df[df.sentence.str.contains('^(Table of Contents)', regex=True)].index)
    df['sentence'] = df.sentence.str.replace("^(Table of Contents )", repl='', flags=re.IGNORECASE, regex=True)
    
    return df #, fur_preprocess_ids_1, fur_preprocess_ids_2, fur_preprocess_ids_3

def fur_preprocess_split_sentences(sent):
    split_sentences = []
    cur_index = 0
    pattern1 = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s)([A-Z])' # sentences endswith "." subsequently has a captital[A-Z]
    match_start_indexes1 = [match.start() for match in re.finditer(pattern1, sent)]
    match_start_indexes = match_start_indexes1
    if -1 in match_start_indexes:
        match_start_indexes.remove(-1)
    for index in match_start_indexes:
        split_sentences.append(sent[cur_index:index])
        cur_index = index+1
    split_sentences.append(sent[cur_index:])
    return split_sentences

def set_spacy_nlp(doc_max_len=10000000):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = doc_max_len
    nlp.disable_pipes('ner')
    return nlp

def sentence_tokenize(text, nlp):
    # input document
    # output sentences, sentencize by spacy
    doc = nlp(text)
    sentences = list(doc.sents)
    sent_list = []
    for sent in sentences:
        sent_list.append(str(sent.text))
    return sent_list

def fur_preprocess_split_long_sentences(sent):
    """
    TODO: what if nothing match
    """
    split_sentences = []
    cur_index = 0
    pattern1 = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s)([A-Z])' # sentences endswith "." subsequently has a captital[A-Z]
    pattern2 = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|;)\s' # sentences endswith ";" and space
    pattern3 = re.compile(r'(\(i\))|(\(ii\))|(\(iii\))|(\(iv\))|(\(v\))|(\(vi\))|(\(vii\))|(\(viii\))|(\(ix\))|(\(x\))|(\(xi\))|(\(xii\))|(\(xiii\))|(\(xiv\))|(\(xv\))|(\(xvi\))|(\(xvii\))|(\(xviii\))|(\(xix\))|(\(xx\))', flags=re.IGNORECASE) # roman numbers
    pattern4 = re.compile(r'(\(1\))|(\(2\))|(\(3\))|(\(4\))|(\(5\))|(\(6\))|(\(7\))|(\(8\))|(\(9\))|(\(10\))|(\(11\))|(\(12\))|(\(13\))|(\(14\))|(\(15\))|(\(16\))|(\(17\))|(\(18\))|(\(19\))|(\(20\))', flags=re.IGNORECASE) # arab numbers
    pattern5 = re.compile(r'(\(a\))|(\(b\))|(\(c\))|(\(d\))|(\(e\))|(\(f\))|(\(g\))|(\(h\))|(\(i\))|(\(j\))', flags=re.IGNORECASE) # abcd
    pattern6 = re.compile(r'(a\))|(b\))|(c\))|(d\))|(e\))|(f\))|(g\))|(h\))|(i\))|(j\))', flags=re.IGNORECASE) # abcd

    match_start_indexes1 = [match.start() for match in re.finditer(pattern1, sent)]
    match_start_indexes2 = [match.start() for match in re.finditer(pattern2, sent)]
    match_start_indexes3 = [match.start()-1 for match in re.finditer(pattern3, sent)]
    match_start_indexes4 = [match.start()-1 for match in re.finditer(pattern4, sent)]
    match_start_indexes5 = [match.start()-1 for match in re.finditer(pattern5, sent)]
    match_start_indexes6 = [match.start()-1 for match in re.finditer(pattern6, sent)]
    
    # if len(match_start_indexes1)!=0:
    #     match_start_indexes = match_start_indexes1
    # elif len(match_start_indexes2)!=0:
    #     match_start_indexes = match_start_indexes2
    # elif len(match_start_indexes3)!=0:
    #     match_start_indexes = match_start_indexes3
    # elif len(match_start_indexes4)!=0:
    #     match_start_indexes = match_start_indexes4
    # elif len(match_start_indexes5)!=0:
    #     match_start_indexes = match_start_indexes5
    # else:
    #     print('No patterns found')
    #     return [sent]
    
    match_start_indexes = match_start_indexes1+match_start_indexes2+match_start_indexes3+match_start_indexes4+match_start_indexes5+match_start_indexes6
    if -1 in match_start_indexes:
        match_start_indexes.remove(-1)
    for index in match_start_indexes:
        split_sentences.append(sent[cur_index:index])
        cur_index = index+1
    split_sentences.append(sent[cur_index:])
    return split_sentences

def fur_preprocess_insert_split_sentences(df, tokenizer, nlp):
    """
    input before split
    """
    # cur_idx = 0
    # new_df = pd.DataFrame()
    # df_split_sentence = pd.DataFrame(columns=['fullID', 'sentence', 'sent_length'])
    # for idx, sent in zip(df[df['sent_length']>256].index, df[df['sent_length']>256].sentence):
    #     # print(idx, df['fullID'].loc[idx])
    #     split_sentences = fur_preprocess_split_long_sentences(sent)
        
    #     df_split_sentence['sentence'] = split_sentences
    #     df_split_sentence['fullID'] = [df['fullID'].loc[idx]]*len(df_split_sentence)
    #     df_split_sentence = get_length(df_split_sentence, tokenizer)
        
    #     new_df = new_df.append(df[cur_idx:idx], ignore_index=True)
    #     cur_idx = idx+1 # skip the long sentence to be slpit
    #     new_df = new_df.append(df_split_sentence, ignore_index=True) # append the split sentences
    #     df_split_sentence = pd.DataFrame(columns=['fullID', 'sentence', 'sent_length']) # reset

    # new_df = new_df.append(df[cur_idx:], ignore_index=True) # append the rest

    cur_idx = 0
    new_df = pd.DataFrame()
    df_split_sentence = pd.DataFrame(columns=['fullID', 'sentence', 'sent_length'])
    for idx, sent in zip(df.index, df.sentence):

        if df.loc[idx, 'sent_length']>256:
            split_sentences = fur_preprocess_split_long_sentences(sent)
            df_split_sentence['sentence'] = split_sentences
            df_split_sentence['fullID'] = [df['fullID'].loc[idx]]*len(df_split_sentence)
            df_split_sentence = get_length(df_split_sentence, tokenizer)
            new_df = new_df.append(df[cur_idx:idx], ignore_index=True)
            cur_idx = idx+1 # skip the long sentence to be slpit
            new_df = new_df.append(df_split_sentence, ignore_index=True) # append the split sentences
            df_split_sentence = pd.DataFrame(columns=['fullID', 'sentence', 'sent_length']) # reset
        else:
            # split_sentences = fur_preprocess_split_sentences(sent)
            split_sentences = sentence_tokenize(sent, nlp)
            df_split_sentence['sentence'] = split_sentences
            df_split_sentence['fullID'] = [df['fullID'].loc[idx]]*len(df_split_sentence)
            df_split_sentence = get_length(df_split_sentence, tokenizer)
            new_df = new_df.append(df[cur_idx:idx], ignore_index=True)
            cur_idx = idx+1 # skip the long sentence to be slpit
            new_df = new_df.append(df_split_sentence, ignore_index=True) # append the split sentences
            df_split_sentence = pd.DataFrame(columns=['fullID', 'sentence', 'sent_length']) # reset

    new_df = new_df.append(df[cur_idx:], ignore_index=True) # append the rest
    return new_df

def id_revise(df):
    # find duplicate id paragraph
    new_df = pd.DataFrame(df)
    id_para_to_be_revised = ['_'.join(fullID.split('_')[:-1])+"_" for fullID in list(set(df[df.duplicated(subset='fullID')]['fullID']))]
    
    # replace all those duplicate paragraph ids with new sentence id
    for id_para in id_para_to_be_revised:
        df1 = df[df['fullID'].str.contains(id_para)]
        for sid in range(len(df1)):
            df1['fullID'].iloc[sid] = f'{id_para}S{sid}'
        new_df.loc[df[df['fullID'].str.contains(id_para)].index] = df1
    return new_df
    
def main(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    file = args.collection
    output = args.output
    sent_output = args.sent_length
    data = read_collections(file)


    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df.columns = ['fullID', 'sentence']
    # df = df[:10000]
    # df = fur_preprocess(df) # replace useless prefix
    df = get_length(df, tokenizer)

    nlp = set_spacy_nlp()
    new_df = fur_preprocess_insert_split_sentences(df, tokenizer, nlp) # split by patterns
    new_id_df = id_revise(new_df) # get id correct
    new_id_df['sentence'] = new_id_df['sentence'].str.replace(r'\s+', ' ', regex=True)
    new_id_df['sentence'] = new_id_df['sentence'].str.strip()

    new_id_df['sent_length'].to_csv(sent_output, header=None, index=False)
    new_id_df = new_id_df.drop(columns=['sent_length'])
    new_id_df.to_csv(output, sep='\t', header=None, index=False, quoting=csv.QUOTE_NONE)

    

if __name__ == '__main__':
    main(args)