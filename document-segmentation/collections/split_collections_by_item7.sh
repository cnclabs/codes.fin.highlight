while read -r cmp
do 
echo $cmp
mkdir /tmp2/cwlin/fintext-new/tmp-collections/all/rand100/item7/by-company/$cmp
grep '_ITEM7_' /tmp2/cwlin/fintext-new/collections/all/rand100/by-company/$cmp/collections.txt > /tmp2/cwlin/fintext-new/collections/all/rand100/item7/by-company/$cmp/collections.txt
done < '/tmp2/cwlin/fintext-new/collections/rand100-CIKS.txt'