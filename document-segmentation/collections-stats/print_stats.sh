# python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100
# python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/prediction/rand100-predict-collection --output_dir /tmp2/cwlin/fintext-new/tmp-collections/prediction
# python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/prediction-256/rand100-predict-collection-256 --output_dir /tmp2/cwlin/fintext-new/tmp-collections/prediction-256
# python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/prediction-512/rand100-predict-collection-512 --output_dir /tmp2/cwlin/fintext-new/tmp-collections/prediction-512

# python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/sent_and_seg_length.csv
# python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/prediction/sent_and_seg_length.csv
# python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/prediction-256/sent_and_seg_length.csv
# python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/prediction-512/sent_and_seg_length.csv

# rand0-100
echo 'rand0-100'
# origin parse collection
echo 'origin parse collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/sent_and_seg_length.csv
# cross-seg predict collection
echo 'cross-seg predict collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction/rand0-100-predict-collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction/sent_and_seg_length.csv

# rand100-200
echo 'rand100-200'
# origin parse collection
echo 'origin parse collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/sent_and_seg_length.csv
# cross-seg predict collection
echo 'cross-seg predict collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction/rand100-200-predict-collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction/sent_and_seg_length.csv

# rand200
echo 'rand200'
# origin parse collection
echo 'origin parse collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand200
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand200/sent_and_seg_length.csv
# cross-seg predict collection
echo 'cross-seg predict collection'
python3 print_stats_tocsv.py --collection /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/rand200-all-item-predict-collections.txt --output_dir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction
python3 print_stats.py --sent_and_seg_length_file /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/sent_and_seg_length.csv