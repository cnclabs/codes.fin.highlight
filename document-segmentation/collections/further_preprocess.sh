task(){
CIK=$(echo $file| cut -d'/' -f 9)
echo /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/by-company/$CIK/test_further_preprocess_collections.txt
python test_further_preprocess.py --collection "$file" --output  /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/by-company/$CIK/test_further_preprocess_collections.txt --sent_length /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/by-company/$CIK/test_sentence_length.txt;
sleep 1;
}

N=4
(
for file in /tmp2/cwlin/fintext-new/collections/8y-item7/rand100/by-company/*/collections.txt; do
   ((i=i%N)); ((i++==0)) && wait
   task "$file" &
done
)