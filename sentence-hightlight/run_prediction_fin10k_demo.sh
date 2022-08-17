export CUDA_VISIBLE_DEVICES=0
BS=16
STEPS=12500
TYPE=esnli-zs-highlighter

for com in 70145 70487 70502 710782 712771 714603 715579 715957 717538 72162 726728 72903 733076 749647 75679 764401 768835 769520 77281 775057 785786 794619 797468 798949 80035 81033 811596 813762 820313 832480 833640 837465 844161 845877 863110 866374 8670 873044 874292 877890 880285 884219 884887 886128 886163 889331 889900 891103 8947 897077 898173 90144 906709 908259 911649 912750 913144 916365 920112 922621 932781 9389 94344 945394 945983 94845 948708 949158 97517 97745;do
    for FILE in data/fin10k-demo/$com/ITEM*.type2;do
    OUTPUT_FILE=${FILE/data/results}
        if [ -s $FILE ]
        then
            python3 inference.py \
              --model_name_or_path checkpoints/${TYPE}/checkpoint-${STEPS}/ \
              --config_name bert-base-uncased \
              --eval_file $FILE \
              --result_json $OUTPUT_FILE.jsonl \
              --remove_unused_columns false \
              --max_seq_length 512 \
              --per_device_train_batch_size $BS \
              --do_eval \
              --prob_aggregate_strategy 'max'
        else
            echo File $FILE empty
            mkdir -p ${OUTPUT_FILE%/*}
            touch $OUTPUT_FILE
            echo Create empty file and skip processing.
        fi
    done
done
