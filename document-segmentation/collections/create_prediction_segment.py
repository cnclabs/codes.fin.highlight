import argparse
import pandas as pd
from collections import OrderedDict
from transformers import BertTokenizer, BertModel
pd.options.mode.chained_assignment = None # ignore false alarm user warning

parser = argparse.ArgumentParser()
parser.add_argument('--prediction', help='predict file') # /tmp2/cwlin/fintext-new/collections/prediction/rand43/ITEM7/rand43-100000
parser.add_argument('--collection', help='collection') # /tmp2/cwlin/fintext-new/collections/rand100/ITEM7/collections.txt
parser.add_argument('--limit_length', help='256, 512') 
parser.add_argument('--thresh', type=float, default=0.5, help='output')
parser.add_argument('--output', help='output') # /tmp2/cwlin/fintext-new/collections/prediction-collection/rand43/ITEM7/rand43-predict-collection
args = parser.parse_args()

def len_sent(tokenizer, text):
    return len(tokenizer.tokenize(text))

def predict_threshold(df, thresh=0.5):
    """
    given df, move probability threshold to create new prediction
    """
    df['pred'].loc[df['prob']>float(thresh)] = 1
    df['pred'].loc[df['prob']<=float(thresh)] = 0
    return df

def get_data(prediction, collection, thresh):
    """
    prediction: fullID \t prediction \t probability
    collection: cik_year_item_pid_sid \t sentence
    return: df -> prediction
    return: data -> collection
    """
    print('Getting data...')
    df = pd.read_csv(prediction, sep='\t', header=None)
    df.columns = ['fullID', 'pred', 'prob']
    if thresh!=None:
        df = predict_threshold(df, thresh=thresh)

    data = dict()
    with open(collection, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            fullID, sentence = line.split('\t')
            cik, year, item, pid, sid = fullID.split('_')
            cik_year_item = f'{cik}_{year}_{item}'
            
            data[f'{cik_year_item}_{pid}_{sid}'] = sentence
    f.close()
    return data, df

def format_bound_to_mass(string, boundary_symbol='1'):
    """
    input: '1001000010'
    output:'352'
    """
    string = '1'+string[1:] # force the first to be breakpoint
    masses = [len(segment) + 1 for segment in string.split(boundary_symbol)][1:]
    return masses

def format_mass_to_position(masses):
    """
    input: '[5,3,5]'
    output: '1111122233333'
    """
    sequence = list()
    for i, mass in enumerate(masses):
        sequence.extend([i] * mass)
    return list(tuple(sequence))

def add_and_revise_position_and_prediction(df):
    # revise nothing
    preds = ''.join(list(str(pred) for pred in df.pred))
    # print('preds', len(preds), 'df', len(df))
    df['position'] = format_mass_to_position(format_bound_to_mass(preds))

    # revise prediction across documents
    df['re_pred'] = df['pred'] # replicate one and replace only across documents predictions with 1
    dfx = df['fullID'].str.split('_', expand=True)[[3,4]] # cik_year_item_para_sent -> cik, year, item, para, sent

    for idx in dfx.loc[(dfx[3]=='P0') & (dfx[4]=='S0')].index:
        df['re_pred'][idx]=1
    print('Revised prediction across documents')

    re_preds = ''.join(list(str(pred) for pred in df.re_pred))
    df['re_position'] = format_mass_to_position(format_bound_to_mass(re_preds))

    # revise prediction across paragraphs
    df['re_re_pred'] = df['re_pred'] # replicate one and replace only across paragraph predictions with 1
    dfx = df['fullID'].str.split('_', expand=True)[[3,4]]

    change_para_index = []
    cur_pidx = '0'
    for idx, pidx in enumerate(dfx[3]):
        pidx = pidx.replace('P', '')
        if cur_pidx!=pidx: # detect change of paragraph
            change_para_index.append(idx)
            cur_pidx = pidx

    for idx in change_para_index: # replace all across paragraph
        df['re_re_pred'][idx]=1

    re_re_preds = ''.join(list(str(pred) for pred in df.re_re_pred))
    df['re_re_position'] = format_mass_to_position(format_bound_to_mass(re_re_preds))
    print('Revised prediction across paragraph')
    return df

def create_new_id_segment(df, data):
    """
    remain orginial PID, and create new code: G, for segments
    for example:
    |ABC|DE| --> |AB|C|D|E|
        P0-S0 ; sentA
        P0-S1 ; sentB
        P0-S2 ; sentC  <--- need to make sure there is no across paragraphs / documents e.g. |AB|CD|E|
        P1-S0 ; sentD
        P1-S1 ; sentE
    --> P0-G0 ; sentA + sentB
    --> P0-G1 ; sentC
    --> P1-G2 ; sentD
    --> P1-G3 ; sentE
    No longer use this
    """
    new_data = OrderedDict()

    cur_ID_CIK_YEAR_ITEM = 0
    for pos in list(set(df.re_re_position)):
        df3 = df[df['re_re_position']==pos]

        # # across documents
        # ID_CIK_YEAR_ITEMs = list([fullID.split('-')[0] for fullID in df3.fullID])
        # if ID_CIK_YEAR_ITEMs[0]!=ID_CIK_YEAR_ITEMs[-1]:
        #     print(df3)

        ID_CIK_YEAR_ITEM = list([fullID.split('-')[0] for fullID in df3.fullID])[0]
        
        if cur_ID_CIK_YEAR_ITEM!=ID_CIK_YEAR_ITEM:
            gid=0
            cur_ID_CIK_YEAR_ITEM = ID_CIK_YEAR_ITEM
        
        pids = [num.replace('P', '') for num in list(set([fullID.split('-')[1] for fullID in df3.fullID]))]
        # if pids[0]!=pids[-1]: # across paragraph
        #     print(df3)

        # new_id = f'{ID_CIK_YEAR_ITEM}_P{pids[0]}-{pids[-1]}_G{gid}' # in across paragraph settings
        new_id = f'{ID_CIK_YEAR_ITEM}_P{pids[0]}_G{gid}' # no longer across paragraph

        new_segment = ''
        for id in df3.fullID:
            new_segment += data[id].strip()
            new_segment += ' '
        new_segment = new_segment.strip()
        gid+=1
        
        new_data[new_id] = new_segment
    return new_data

def replace_new_id_segment(df, data, position_col='re_long_seg_position'):
    """
    replace PID with new PID
    """
    new_data = OrderedDict()
    cur_ID_CIK_YEAR_ITEM = ''
    for pos in sorted(list(set(df[position_col]))):
        df3 = df[df[position_col]==pos]
        ID_CIK_YEAR_ITEM = list(["_".join(fullID.split('_')[:3]) for fullID in df3.fullID])[0]
        # detect change item
        if cur_ID_CIK_YEAR_ITEM!=ID_CIK_YEAR_ITEM:
            pid=0
            cur_ID_CIK_YEAR_ITEM = ID_CIK_YEAR_ITEM
        for i in range(len(df3)):
            new_id = f'{ID_CIK_YEAR_ITEM}_P{pid}_S{i}'
            new_data[new_id] = data[df3.iloc[i]['fullID']]
        pid+=1

    return new_data

def write_new_id_segment(output, new_data):
    with open(output, 'w') as f:
        for id, segment in new_data.items():
            f.write(f'{id}\t{segment}\n')
    f.close()
    print('Done writing!')

def break_long_segment(df, data, tokenizer, limit_length):
    df['is_long_seg'] = [0]*len(df)
    df[['re_long_seg_pred', 're_long_seg_position']] = df[['re_re_pred', 're_re_position']]
    loop = 0
    long_seg_exists = True
    while long_seg_exists:
        if loop==0:
            # first time: check all positions segment length
            positions = list(set(df.re_long_seg_position))
            print(f'Original segment count:\t\t{len(positions)}')
        else:
            # check only previous long segments
            positions = list(set(df[df['is_long_seg']==1].re_long_seg_position))
            print(f'Updated long segment count:\t\t{len(positions)}')
        
        # check and update long segments
        df['is_long_seg'] = [0]*len(df) # reset 0
        for pos in positions:
            df3 = df[df['re_long_seg_position']==pos]
            seg = " ".join([data[fullID] for fullID in df3.fullID])
            if len_sent(tokenizer, seg)>limit_length:
                if len(df3)>1:
                    df['is_long_seg'].loc[df3.index]=1
                else: # sentence itself is too long
                    print(df3['fullID'].values, len_sent(tokenizer, seg),'sentence itself is too long')
                    pass
        if loop==0:
            positions = list(set(df[df['is_long_seg']==1].re_long_seg_position))
            print(f'Original long segment count:\t\t{len(positions)}')

        # break long segment
        for pos in sorted(list(set(df[df['is_long_seg']==1].re_long_seg_position))):
            df3 = df[df['re_long_seg_position']==pos]

            # find maxprob in segment and create new seg
            df4 = df3[df3['re_long_seg_pred']!=1] # disgard first sentence in segment
            maxprob_idx = df4.prob.idxmax()
            df['re_long_seg_pred'].loc[[maxprob_idx]] = 1
        preds = ''.join(list(str(pred) for pred in df.re_long_seg_pred))
        df['re_long_seg_position'] = format_mass_to_position(format_bound_to_mass(preds))

        loop+=1
        long_seg_exists = True if len(df[df['is_long_seg']==1])>0 else False
        print(f'loop:{loop}')
    return df

def main(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data, df = get_data(args.prediction, args.collection, args.thresh)
    df = add_and_revise_position_and_prediction(df)

    if args.limit_length!=None:
        print(f'length limitation: {args.limit_length}')
        df = break_long_segment(df,data,tokenizer, int(args.limit_length))
        new_id_segment_data = replace_new_id_segment(df, data, position_col='re_long_seg_position')
    else:
        print('No length limit')
        new_id_segment_data = replace_new_id_segment(df, data, position_col='re_re_position')

    write_new_id_segment(args.output, new_id_segment_data)

if __name__ == '__main__':
    main(args)
