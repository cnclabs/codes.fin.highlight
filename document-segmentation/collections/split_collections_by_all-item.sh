# while read -r item
# do
# echo _"$item"_
# mkdir /tmp2/cwlin/fintext-new/test-collections/rand200/by-item/$item
# grep -E _"$item"_ /tmp2/cwlin/fintext-new/test-collections/rand200/rand200-all-item-collections.txt > /tmp2/cwlin/fintext-new/test-collections/rand200/by-item/$item/rand200_collections.txt
# done < '/tmp2/cwlin/fintext-new/codes.fin.contextualized/document-segmentation/parsed-collections/list-of-items.txt'

# while read -r item
# do
# echo _"$item"_
# mkdir /tmp2/cwlin/fintext-new/test-collections/rand200/predict-all/prediction/by-item/$item
# grep -E _"$item"_ /tmp2/cwlin/fintext-new/test-collections/rand200/predict-all/prediction/rand200-predict-collection > /tmp2/cwlin/fintext-new/test-collections/rand200/predict-all/prediction/by-item/$item/rand200_collections.txt
# done < '/tmp2/cwlin/fintext-new/codes.fin.contextualized/document-segmentation/parsed-collections/list-of-items.txt'

# origin
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/by-item
while read -r item
do
echo _"$item"_
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/by-item/$item
grep -E _"$item"_ /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections.txt > /tmp2/cwlin/fintext-new/tmp-collections/rand200/by-item/$item/rand200_collections.txt
done < '/tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/parse/list-of-items.txt'

# cross-seg predict
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/by-item
while read -r item
do
echo _"$item"_
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/by-item/$item
grep -E _"$item"_ /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/rand200-all-item-predict-collections.txt > /tmp2/cwlin/fintext-new/tmp-collections/rand200/predict-all/prediction/by-item/$item/rand200_collections.txt
done < '/tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/parse/list-of-items.txt'