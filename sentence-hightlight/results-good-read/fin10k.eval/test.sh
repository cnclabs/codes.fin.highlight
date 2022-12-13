
for setting in from-scratch further-fine-tune esnli-zs-highlighter further-finetune-sl-smooth-2;do
    touch $setting.rp.py
    cat type2/fin10k.eval.type2.results-further-finetune.log | grep R-Prec >> $setting.rp.py
    cat type1.easy/fin10k.eval.type1.easy.results-further-finetune.log | grep R-Prec >> $setting.rp.py
    cat type1.hard/fin10k.eval.type1.hard.results-further-finetune.log | grep R-Prec >> $setting.rp.py

    touch $setting.pcc.py
    cat type2/fin10k.eval.type2.results-further-finetune.log | grep "Pearson:" >> $setting.pcc.py
    cat type1.easy/fin10k.eval.type1.easy.results-further-finetune.log | grep "Pearson:" >> $setting.pcc.py
    cat type1.hard/fin10k.eval.type1.hard.results-further-finetune.log | grep "Pearson:" >> $setting.pcc.py
done


