# get all items from rand200 companies
all_cmp_pat=""
while read -r cmp
do 
all_cmp_pat+=^"$cmp"_\|
done < '/tmp2/cwlin/fintext-new/codes/codes.fin.highlight/document-segmentation/collections/rand200-CIKS.txt'

all_cmp_pat=${all_cmp_pat::-1}

echo $all_cmp_pat
grep -E $all_cmp_pat /tmp2/cwlin/fintext-new/tmp-collections/all/collections.txt > /tmp2/cwlin/fintext-new/tmp-collections/rand200/rand200-all-item-collections.txt
