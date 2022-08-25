wget -q https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_1.csv \
    -O esnli_train.csv
wget -q https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_train_2.csv \
    -O esnli_train_2.csv
wget -q https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_dev.csv \
    -O esnli_dev.csv
wget -q https://github.com/OanaMariaCamburu/e-SNLI/raw/master/dataset/esnli_test.csv \
    -O esnli_test.csv
tail -n +2 esnli_train_2.csv >> esnli_train.csv
