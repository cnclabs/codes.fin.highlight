folder=$1
metric=$2
annotator=$3
for file in $1/*results-further-finetune*.log$annotator;do
    echo $file 
    tail $file | grep "Mean $metric"
    echo -e
done
