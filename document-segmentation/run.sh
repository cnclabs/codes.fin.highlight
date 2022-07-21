# extract raw data
python3 /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/extract-raw-data/extract_data.py \
--raw_data_path /tmp2/cwlin/fintext-new/raw-data \
--file_list_10k /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/extract-raw-data/10k-file-list.txt \
--output_10k_file_list /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/extract-raw-data/10k-file-list.txt \
--output_8y_10k_file_list /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/extract-raw-data/8y-10k-company-file-list.txt

# cd /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/parse

mkdir /tmp2/cwlin/fintext-new/tmp-collections
mkdir /tmp2/cwlin/fintext-new/tmp-collections/all
mkdir /tmp2/cwlin/fintext-new/tmp-collections/8y-item7
mkdir /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch

mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200

mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction

# if length limit needed 
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction-256
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction-512
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction-256
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction-512
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction-256
# mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction-512

# parse
for i in {0..12}
do
python3 parse_main.py --company_file_list /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/extract-raw-data/8y-10k-company-file-list.txt \
--batch_number $i \
--batch_size 300 \
--path_output_dir /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch \
--abnormal_dir /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch >> /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch/logs-$i &
done

# combine all batch
cat /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch/collections-* >> /tmp2/cwlin/fintext-new/tmp-collections/all/collections.txt
cat /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch/abnormal-cmp-* >> /tmp2/cwlin/fintext-new/tmp-collections/all/abnormal-cmp.txt
cat /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch/logs-* >> /tmp2/cwlin/fintext-new/tmp-collections/all/logs

# rm -r /tmp2/cwlin/fintext-new/tmp-collections/parsed-batch # remove the batch

# filter collection
cd /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/collections

# generate random company
# python gen_random_company.py
# path here
    # /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/collections
    # ./rand100-CIKs.txt
    # ./rand100-200-CIKs.txt
    # ./rand200-CIKs.txt

# for rand200 all items (not just item7)
bash get_all_item_rand200.sh


# for rand100, rand100-200, and item7 only
# this filter to 8y-item7-collections and list the companies that item7 has at least 5 sentences (see logs.company.statistics for detail)
bash filter_collection.sh >> logs.company.statistics

# split by company so that we can use rand0-100, rand100-200 etc. (origin-> 2293, split->rand100, rand100-200)
bash split_collections_by_company.sh

# combine above selected rand100 or rand100-200 companies collections
bash combine-collections.sh

# 
cd /tmp2/jhju/temp/document-segmentation/
# bash run_prediction_fin10k.sh <path-of-your-parsed-txt>
bash run_prediction_fin10k.sh /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/collections.txt
bash run_prediction_fin10k.sh /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/collections.txt
bash run_prediction_fin10k.sh /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections.txt

cd /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/collections
# create prediction segment of rand100, rand100-200, and rand200-all-item
bash create_prediction_segment.sh

# print statistics
cd /tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/collections-stats
bash print_stats.sh >> logs.collections.statistics

# split collections by all item
bash split_collections_by_all-item.sh