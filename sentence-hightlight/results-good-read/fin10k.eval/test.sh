touch rp.py
touch pcc.py

for setting in esnli-zs-highlighter further-finetune-sl-smooth-2;do
    echo "rp['$setting'] = ["  >> rp.py
    cat type2/fin10k.eval.type2.results-$setting.log | grep R-Prec | cut -c 11- >> rp.py
    cat type1.easy/fin10k.eval.type1.easy.results-$setting.log | grep R-Prec | cut -c 11- >> rp.py
    cat type1.hard/fin10k.eval.type1.hard.results-$setting.log | grep R-Prec | cut -c 11- >> rp.py
    echo "]"  >> rp.py

    echo "pcc['$setting'] = ["  >> pcc.py
    cat type2/fin10k.eval.type2.results-$setting.log | grep "Pearson: " | cut -c 13- >> pcc.py
    cat type1.easy/fin10k.eval.type1.easy.results-$setting.log | grep "Pearson: " | cut -c 13- >> pcc.py
    cat type1.hard/fin10k.eval.type1.hard.results-$setting.log | grep "Pearson: " | cut -c 13- >> pcc.py
    echo "]"  >> pcc.py
done

