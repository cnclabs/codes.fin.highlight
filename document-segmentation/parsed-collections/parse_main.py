import argparse
from test_parse_utils import load_cmp_dict, identify_item, identify_paragraph, parsing, get_item_paragraph, clean_item_paragraph, sentencize_item_paragraph, remove_empty_keys, reset_paragraph_sent_num, set_spacy_nlp
from collections import OrderedDict 

parser = argparse.ArgumentParser()
parser.add_argument("--company_file_list", type=str)
parser.add_argument("--batch_number", type=str)
parser.add_argument("--batch_size", type=str)
parser.add_argument("--path_output_dir", type=str)
parser.add_argument("--abnormal_dir", type=str)
args = parser.parse_args()

cmp_fp = args.company_file_list
cmp_dict = load_cmp_dict(cmp_fp)
batch_size = int(args.batch_size)
nlp = set_spacy_nlp()
output = open(args.path_output_dir+f'/collections-{args.batch_number}.txt', 'w', encoding='utf-8')

# cmp_fp = '/tmp2/cwlin/fintext-new/document-segmentation/extract-raw-data/8y-10k-company-file-list.txt'
# with open('/tmp2/cwlin/fintext-new/test/empty-cmp-list', 'r') as f:
#     fp_list = [line.replace('\n', '') for line in f.readlines()]
# f.close()

# fp_list = ['31791']
abnormal_cmp = []
for i, (cmp, fp_list) in enumerate(cmp_dict.items()):
    if batch_size*int(args.batch_number) <= i < batch_size*(int(args.batch_number)+1):
        for f in fp_list:
            cik = f.split('/')[-1].split('_')[4]
            year = f.split('/')[-1].split('_')[5].split('-')[1]
            print(f'Parsing... {cik}-{year}')
            all_doc_raw_text = identify_item(f)
            data = identify_paragraph(all_doc_raw_text)
            document_items = parsing(data)
            document_items_paragraphs_sentences = OrderedDict()
            for item_title in document_items.keys():
                item_paragraph = get_item_paragraph(document_items, item_title)
                # item_paragraph, drop_li = clean_item_paragraph(item_paragraph)
                item_paragraph_dict = sentencize_item_paragraph(item_paragraph, nlp)
                item_paragraph_dict = remove_empty_keys(item_paragraph_dict)
                item_paragraph_dict = reset_paragraph_sent_num(item_paragraph_dict)
                document_items_paragraphs_sentences[item_title] = item_paragraph_dict

            print('Get final number of items: ', len(document_items_paragraphs_sentences))
            
            if len(document_items_paragraphs_sentences)==0:
                abnormal_cmp.append(f)


            for item, paragraph_dicts in document_items_paragraphs_sentences.items():
                for paragraph_num, sentence_dicts in paragraph_dicts.items():
                    for sent_num, sentence in sentence_dicts.items():
                        output.write(f'{cik}_{year}_{item}_P{paragraph_num}_S{sent_num}\t{sentence}\n')

output.close()

if args.abnormal_dir!=None:
    with open(f'{args.abnormal_dir}/abnormal-cmp-{args.batch_number}', 'w') as f:
        for cmp in abnormal_cmp:
            f.write(f'{cmp}\n')
        f.close()
    print(f'Number of abnormal companies: ', len(abnormal_cmp))