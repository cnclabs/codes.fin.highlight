# #!/bin/bash
# for i in {0..15}
# do 
#     python parse_main.py \
#     --company_file_list /tmp2/cwlin/fintext/codes.fin.contextualized/document_segmentation/regex/8y-10k-company-file-list.txt \
#     --batch_number $i \
#     --batch_size 240 \
#     --path_output_dir /tmp2/cwlin/fintext/data/parsed_8y_10k_cmp/batch-16-cmp-240 &
# done

for i in {0..12}
do
python3 parse_main.py --company_file_list /tmp2/cwlin/fintext-new/document-segmentation/extract-raw-data/8y-10k-company-file-list.txt \
--batch_number $i \
--batch_size 300 \
--path_output_dir /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch \
--abnormal_dir /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch >> /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch/logs-$i &
done