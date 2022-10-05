folder=$1
annotator=$2
for file in $1/*results-further-finetune*.log.$annotator;do
    echo $file 
    # tail $file | grep "Mean Pearson"
    tail $file | grep "Mean RPrecision"
    echo -e
done
