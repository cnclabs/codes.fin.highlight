import os
import random
random.seed(2022)
# path = '/tmp2/cwlin/fintext-new/collections/8y-item7/company'
path = '/tmp2/cwlin/fintext-new/tmp-collections/8y-item7-company-more_than_5_sentences.txt'
first_rand100_path = '/tmp2/cwlin/fintext-new/collections/rand100-CIKS.txt'

with open(path, 'r') as f:
    all_company = [line.replace('\n', '') for line in f.readlines()]
f.close()

with open(first_rand100_path, 'r') as f:
    first_rand100_company = [line.replace('\n', '') for line in f.readlines()]
f.close()

company = list(set(all_company)-set(first_rand100_company))

# all_company = os.listdir(path)
rand_cmp = random.sample(company, 100)

with open('/tmp2/cwlin/fintext-new/tmp-collections/rand100-200-CIKS.txt', 'w') as f:
    for cmp in rand_cmp:
        f.write(f'{cmp}\n')
f.close()