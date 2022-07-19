# Type1 contain all one (highlight all)
# Type0 contain all zero (no highlight)
# Type2 contain all one (highlight all) --> (highlight partial)
echo "Createing data for bert...(demo)" > fin10k.demo.dat 

for FILE in fin10k-demo/data/*/*;do
    echo Convert $FILE into jsonl format
    python3 scripts/create_fin10k_data_new.py \
        -input $FILE \
        -output ${FILE/data\//} \
        -type ${FILE##*type} \
        -nosep \
        -format jsonl \
        -model_type bert >> fin10k.demo.dat
done
