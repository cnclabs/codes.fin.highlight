# for file in "/tmp2/cwlin/fintext-new/collections/8y-item7/test/*/further*.txt"
# do
#     cat $file >> /tmp2/cwlin/fintext-new/collections/8y-item7/test/test-collection
# done

# while read -r line;
# do
#     cat $line >> /tmp2/cwlin/fintext-new/collections/8y-item7/company/rand100-collections.txt;
# done < "/tmp2/cwlin/fintext-new/collections/rand100-CIKS-filelist.txt"

# while read -r line;
# do
#     cat $line >> /tmp2/cwlin/fintext-new/collections/prediction/rand100/rand100-predict-collections.txt;
# done < "/tmp2/cwlin/fintext-new/collections/prediction/rand100/rand100-prediction-CIKS-filelist.txt"

for file in "/tmp2/cwlin/fintext-new/tmp-collections/rand100-200/company/*/collections.txt"
do
    cat $file >> /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/collections.txt
done