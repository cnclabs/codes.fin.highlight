import re
import spacy
from collections import OrderedDict

def load_cmp_dict(fp):
    cmp_dict = {}
    with open(fp, 'r') as f:
        data = f.readlines()
        for line in data:
            line = line.replace('\n', '')
            cmp = line.split('_')[4]
            if cmp not in cmp_dict.keys():
                cmp_dict[cmp] = [line]
            else:
                cmp_dict[cmp].append(line)
    f.close()
    return cmp_dict

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

def normalized(text):
    text = text.strip()
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def identify_item(f):
    # return raw text of the whole document
    data = open(f, 'r')
    data = data.readlines()
    p_item = re.compile(r'^Item \d+\w*', re.IGNORECASE)
    newlines = []
    for line in data:
        line = line.lstrip(' ')
        # Item 1A ---become---> |Item 1A|
        # In case of item pattern occurs in the paragrpah, instead of the start of the sentence
        if re.search(p_item, line) is not None:
            line = normalized(line)
            line = re.sub(p_item, f"|{re.search(p_item, line).group(0)}|", line)
            # print(line)
        newlines.append(line)
    return " ".join(newlines)
    
def identify_paragraph(text):
    ###
    # aa\n \nbb\n   \ncc
    # ------ become ------
    # aa\n=== p{i} === \nbb\n=== p{i+1} ===\ncc
    ###
    p_paragraph = re.compile(r'\n\s*\n')
    m = re.findall(p_paragraph, text)
    for i in range(len(m)):
        text = re.sub(p_paragraph, f'\n=== p{i+1} ===\n', text, 1)
    
    return text

def parsing(data):
    item_titles = ['ITEM1', 'ITEM1A', 'ITEM1B', 'ITEM2', 'ITEM3', 'ITEM4', \
                'ITEM5', 'ITEM6', 'ITEM7', 'ITEM7A', 'ITEM8', 'ITEM9', 'ITEM9A', 'ITEM9B', \
                'ITEM10', 'ITEM11', 'ITEM12', 'ITEM13', 'ITEM14', \
                'ITEM15']
    item_flag = -1
    p_item = re.compile(r'\|Item \d+\w*\|', flags=re.IGNORECASE)
    
    document_items = OrderedDict()
    list_data = data.split('\n')
    while len(list_data)>0:
        line = list_data.pop(0)
        if re.search(p_item, line) is not None:
            item_flag = 1
            item_title = re.findall(p_item, line)[0]
            item_title = re.sub(r"\s+", "", item_title)
            item_title = item_title.replace('|', "").upper()
            document_items[item_title] = []

        if (item_flag==1) & (line!=''):
            document_items[item_title].append(line)
    print('get number of items before filter:', len(document_items))
    
    # remove item other than 20 options
    doc_items = {}
    for item in document_items:
        if item in item_titles:
            doc_items[item] = document_items[item]

    return doc_items

def get_item_paragraph(doc_item, item_title):
    paragraph_num = 0
    item_paragraph = [[]]
    p_paragraph = re.compile(r'=== p(\d)+ ===')
    drop_li = []
    for line in doc_item[item_title]:
        line = re.sub(r'\s+', ' ', line).strip()
        if line.startswith('|') or line.upper().startswith('TABLE OF CONTENTS') or (len(line.split(' '))<=1 and (line.isdigit())):
            drop_li.append(line)
        else:

            if re.search(p_paragraph, line) is not None:
                paragraph_num+=1
                item_paragraph.append([])
            else:
                line = normalized(line)
                item_paragraph[paragraph_num].append(line)

    return item_paragraph

def remove_empty_keys(org_dict):
    new_dict = {k: v for k, v in org_dict.items() if v}
    return new_dict

def reset_paragraph_sent_num(org_dict):
    new_dict = {}
    para_num, sent_num = 0, 0
    for k , para in org_dict.items():
        new_dict[para_num] = {}
        for v , sent in para.items():
            sent = sent.replace('|', ' ')
            sent = re.sub(r'\s+', ' ', sent)
            # print(f'{para_num}-{sent_num}\t{sent}')
            new_dict[para_num][sent_num] = sent.lstrip(' ')
            sent_num += 1
        para_num += 1
        sent_num = 0
    return new_dict

def get_collections_cmp_list(fp='/tmp2/cwlin/fintext/data/complete_8y_10k_cmp/collections.txt'):
    with open(fp, 'r') as f:
        data = [line.replace('\n', '') for line in f.readlines()]
    
    crnt_cik, crnt_yr, crnt_item, crnt_para, crnt_sent = data[0].split('_')
    cmp_list = [crnt_cik]
    for line in data:
        sentID, sent = line.split('\t')
        id_cik, id_yr, id_item, id_para, id_sent = sentID.split('_')
        if crnt_cik!=id_cik:
            cmp_list.append(id_cik)
            crnt_cik = id_cik
    return cmp_list

def parse_sic(fp):
    
    # file path
    with open(fp, 'r') as f:
        text = f.read()
    
    # STANDARD INDUSTRIAL CLASSIFICATION:	 [6221]
    p_sic = re.compile(r'STANDARD INDUSTRIAL CLASSIFICATION:.*[\d+]\]')
    sic = re.search(p_sic, text)
    try:
        sic_code = sic.group(0)[-5:-1]
        # print(sic_code)
        return sic_code
    except:
        print(f'Parse ERROR. {fp}')
        return ''

def clean_item_paragraph(item_paragraph):
    drop_li = [] #things to drop, not as prefix
    new_item_paragraph = []
    for i, para_sentences in enumerate(item_paragraph):
        new_item_paragraph.append([])
        for sent in para_sentences:
            sent = re.sub(r'\s+', ' ', sent).strip()
            # if sent.startswith('|') or sent.upper().startswith('TABLE OF CONTENTS') or len(sent.split(' '))<=1:
            if sent.startswith('|') or sent.upper().startswith('TABLE OF CONTENTS') or (len(sent.split(' '))<=1 and (sent.isdigit())):
                drop_li.append(sent)
                
            else:
                new_item_paragraph[i].append(sent)
    return new_item_paragraph, drop_li

def strip_abnormal(text):
    text = re.sub(r'\|', '', text)
    text = re.sub('/s/', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text
    
def sentencize_item_paragraph(item_paragraph, nlp):
    item_paragraph_dict = {}
    para_num, sent_num = 0, 0
    for para_lines in item_paragraph:
        paragraph = ' '.join(para_lines)
        if paragraph!='':
            item_paragraph_dict[para_num] = {}
            sentences = sentence_tokenize(paragraph, nlp)
            for sent in sentences:
                if (len(sent.split()) > 5) and (sent.endswith('.')):
                    sent = strip_abnormal(sent)
                    item_paragraph_dict[para_num][sent_num] = sent
                    sent_num += 1

            para_num+=1
    return item_paragraph_dict