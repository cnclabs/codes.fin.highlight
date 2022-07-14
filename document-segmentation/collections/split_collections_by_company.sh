while read -r cmp
do 
echo $cmp
mkdir /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/company/$cmp
grep "^"$cmp"_" /tmp2/cwlin/fintext-new/tmp-collections/8y-item7-collections.txt > /tmp2/cwlin/fintext-new/tmp-collections/rand100-200/company/$cmp/collections.txt
done < '/tmp2/cwlin/fintext-new/tmp-collections/rand100-200-CIKS.txt'