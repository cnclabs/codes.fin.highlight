# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/collections/prediction/rand43/ITEM7/rand43-100000 \
# --collection /tmp2/cwlin/fintext-new/collections/rand100/ITEM7/collections.txt \
# --output /tmp2/cwlin/fintext-new/collections/prediction-collection/rand43/ITEM7/rand43-predict-collection

# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/collections/prediction/rand43/ITEM7/rand43-100000 \
# --collection /tmp2/cwlin/fintext-new/collections/rand100/ITEM7/collections.txt \
# --output /tmp2/cwlin/fintext-new/collections/prediction-collection/rand43/ITEM7/test-rand43-predict-collection

# python create_prediction_segment.py \
# --prediction /tmp2/jhju/temp/document-segmentation/prediction/rand100_sentences-100000 \
# --collection /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/collections.txt \
# --output /tmp2/cwlin/fintext-new/collections/prediction-collection/8y-item7/rand100/rand100-predict-collection > logs.0609

# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/collections/prediction/rand100/8y_item7_further_preprocess_collections-100000 \
# --collection /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/8y_item7_further_preprocess_collections.txt \
# --output /tmp2/cwlin/fintext-new/collections/prediction-collection/8y-item7/rand100/rand100-predict-collection > logs.0619

# keep long segment
# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections.txt \
# --output /tmp2/cwlin/fintext-new/tmp-collections/prediction/rand100-predict-collection

# # break long segment 256
# python /tmp2/cwlin/fintext-new/collections/create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections.txt \
# --output /tmp2/cwlin/fintext-new/tmp-collections/prediction-256/rand100-predict-collection-256 > logs.0625.create_prediction_segment

# # break long segment 512
# python /tmp2/cwlin/fintext-new/collections/create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections/8y-item7/rand100/rand100_ITEM7_collections.txt \
# --limit_length 512 \
# --output /tmp2/cwlin/fintext-new/tmp-collections/prediction-512/rand100-predict-collection-512 > logs.0626.create_prediction_segment


##############################
# keep long segment
# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections.txt \
# --limit_length 999999999 \
# --output /tmp2/cwlin/fintext-new/tmp-collections-2/prediction/rand100-200-predict-collection

# # break long segment 256
# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections.txt \
# --limit_length 256 \
# --output /tmp2/cwlin/fintext-new/tmp-collections-2/prediction/rand100-200-predict-collection-256

# # break long segment 512
# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections-100000 \
# --collection /tmp2/cwlin/fintext-new/tmp-collections-2/rand100-200/collections.txt \
# --limit_length 512 \
# --output /tmp2/cwlin/fintext-new/tmp-collections-2/prediction/rand100-200-predict-collection-512

# # rand200
# python create_prediction_segment.py \
# --prediction /tmp2/cwlin/fintext-new/test-collections/rand200/rand200-all-item-collections-100000 \
# --collection /tmp2/cwlin/fintext-new/test-collections/rand200/rand200-all-item-collections.txt \
# --output /tmp2/cwlin/fintext-new/test-collections/rand200/predict-all/prediction/rand200-predict-collection


# rand0-100
python create_prediction_segment.py \
--prediction /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/collections-100000 \
--collection /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/collections.txt \
--output /tmp2/cwlin/fintext-new/tmp-collections/rand0-100/predict-all/prediction/rand0-100-predict-collections.txt

# rand100-200
python create_prediction_segment.py \
--prediction /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/collections-100000 \
--collection /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/collections.txt \
--output /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/predict-all/prediction/rand100-200-predict-collections.txt

# rand200
python create_prediction_segment.py \
--prediction /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections-100000 \
--collection /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections.txt \
--output /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/rand200-all-item-predict-collections.txt