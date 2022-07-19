# type0
# for FILE in ../data/fin10k-demo/*/ITEM*.type0;do
#     echo Aggregating prediction ${results##*/}
#
#     COM_FILE=${FILE##*fin10k-demo/}
#
#     python3 hl_aggregate.py \
#       -pred ${FILE} \
#       -out /tmp2/fin10k/demo/$COM_FILE \
#       -thres -1  > log
# done
python3 hl_aggregate.py \
  -pred ../data/fin10k-demo/1001250/ITEM15.type0 \
  -out /tmp2/fin10k/demo/1001250/ITEM15.type0 \
  -hl_on_a \
  -thres -1
