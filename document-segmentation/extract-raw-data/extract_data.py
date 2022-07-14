import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data_path", type=str, help='path to folders 2011-2018')
parser.add_argument("--file_list_10k", type=str, help='path to 10k file list')

parser.add_argument("--output_10k_file_list", type=str, help='output a file of all the file paths to 10K files')
parser.add_argument("--output_8y_10k_file_list", type=str, help='output a file of all the file paths to 10K files, companies must have 8 years files')
args = parser.parse_args()

def write_10K_fps(data_paths, output_path):
    out = open(f'{output_path}/10k-file-list.txt', 'w')
    # all_data_dict = {}
    for path in data_paths:
        print(path)
        # all_data_dict[f'{path[-4:]}'] = {} # year as key
        for i in range(1,5):
            QTR_path = path+f'/QTR{i}/'
            # all_data_dict[f'{path[-4:]}'][f'QTR{i}'] = []
            for f in os.listdir(QTR_path):
                if f.split('_')[1]=='10-K':
                    # all_data_dict[f'{path[-4:]}'][f'QTR{i}'].append(f)
                    out.write(QTR_path+f+'\n')
    out.close()
    return f'Done writing 10k-file-list.txt at {output_path}'

def get_company_year(files):
    company_year = {}
    for fp in files:
        f = fp.split('/')[-1]
        cmp = f.split('_')[4]
        yr = f.split('_')[5].split('-')[1]
        if cmp not in company_year.keys():
            company_year[cmp] = [(yr, fp)]
        else:
            company_year[cmp].append((yr, fp))
    return company_year

def get_company_year(files):
    company_year = {}
    for fp in files:
        f = fp.split('/')[-1]
        cmp = f.split('_')[4]
        yr = f.split('_')[5].split('-')[1]
        if cmp not in company_year.keys():
            company_year[cmp] = [(yr, fp)]
        else:
            company_year[cmp].append((yr, fp))
    return company_year

def get_cmp_list(company_year, k):
    num_cmp, num_cmps, num_cmp_all, num_all_cmp = 0, 0, 0, 0
    cmp_f_list, cmps_f_list, cmp_all_f_list = [], [], []

    for cmp, f_list in company_year.items():

        num_all_cmp += 1
        if (len(f_list)==k): # for company that has reports from 11-15
            num_cmp+=1
            cmp_f_list += [f[1] for f in f_list]

        if len(f_list)>k: # for companies that has reports from 11-15, plus others 
            cmps_f_list += [f[1] for f in f_list]
            num_cmps+=1

        if (len(f_list)>=k):
            # cmp_all_f_list += [f[1] for f in f_list]
            
            cmp_yr_key = {}
            cmp_fp = []
            
            for (yr, fp) in f_list:
                if yr not in cmp_yr_key.keys():
                    f_version = fp.split('_')[-1].replace('.txt', '')
                    cmp_yr_key[yr] = (f_version, fp)
                    cmp_fp.append(fp)
                else:
                    f_version = fp.split('_')[-1].replace('.txt', '')

                    if int(f_version) > int(cmp_yr_key[yr][0]): # only select latest version
                        # print(f'Current: {f_version}; Existed: {cmp_yr_key[yr]}')
                        # cmp_all_f_list[cmp_all_f_list.index(cmp_yr_key[yr][1])] = fp # locate old version index, replace with the latest version file
                        cmp_fp[cmp_fp.index(cmp_yr_key[yr][1])] = fp
                        cmp_yr_key[yr] = (f_version, fp) # update cmp yr key
                        
            if len(cmp_fp)==k:
                num_cmp_all+=1
                cmp_all_f_list+=cmp_fp

    print(f'Number of company exactly {k} files: {num_cmp}')
    print(f'Number of company that has more than {k} files: {num_cmps}')
    print(f'Number of company that has exactly {k} years file: {num_cmp_all}')
    print(f'Number of company: {num_all_cmp}')

    return cmp_f_list, cmps_f_list, cmp_all_f_list

def main(args):
    data_paths = [f'{args.raw_data_path}/201{i}' for i in range(1,9)]
    if args.output_10k_file_list!=None:
        write_10K_fps(data_paths, args.output_10k_file_list)

    with open(f'{args.file_list_10k}', 'r') as f:
        files = [line.replace('\n', '') for line in f.readlines()]
    f.close()

    company_year = get_company_year(files)
    cmp_f_list, cmps_f_list, cmp_all_f_list = get_cmp_list(company_year, 8)

    with open(f'{args.output_8y_10k_file_list}/8y-10k-company-file-list.txt', 'w') as f:
        for fp in cmp_all_f_list:
            f.write(f'{fp}\n')

    f.close()

if __name__ == '__main__':
    main(args)