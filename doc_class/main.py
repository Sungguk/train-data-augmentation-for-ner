from rule_modules import Replace, Insert, Delete, Move, Replace_ENT
from data_handling_for_heuristic import *
import argparse
import numpy as np
import sys
import pandas as pd


# parameter
#arg_str = ' '.join(sys.argv[1:])
#multiple_n = int(arg_str) 

parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
parser.add_argument('--ner', action='store_true')
parser.add_argument('--doc', action='store_true')




####################################################
#""" for Document Classification """
####################################################
if parser.parse_args().doc == True:

    # Load Origianl Data
    DATASET = 'sst'
    raw_data, label_data = load_csv(DATASET)

    print('\n********** Data Augmentation **********')
    # initialization
    total_new_data = []
    total_new_label = []

    replace = Replace('doc')

    data_rep, label_rep, n_rep = replace.do([raw_data, label_data])    
    
    df = pd.DataFrame(data={"col1": label_rep, "col2": data_rep})
    df.to_csv("./aug_dev_data.csv", sep=',',index=False, header=False)
    


####################################################
#""" for Named Entity Recognition """
####################################################
elif parser.parse_args().ner == True:

    print('\n********** Prepare Dataset **********')

    # Load Origianl Data
    read_file_path = 'dataset/conll2003/train.txt'
    raw_data, label_data = load_conll2003(read_file_path)
    
    ori_num = len(raw_data)
    tot_new_num = 0

    # Data Filter 1 (YES entry & YES sentence form) - 5143
    sent_raw1, sent_label1 = filtering_noENT_sentFORM(raw_data, label_data)

    # Data Filter 2 (YES entry) - 11132
    sent_raw2, sent_label2 = filtering_none_entity(raw_data, label_data)

    print('START: the number of original data: {}'.format(ori_num))

    ###############################
    """ Data Augmentation Start """
    ###############################
    print('\n********** Data Augmentation **********')
    # initialization
    total_new_data = []
    total_new_label = []

    replace = Replace()
    #replace_ent = Replace_ENT()
    insert = Insert()
    #delete = Delete()
    #move = Move()

    #print(filtering_clauses(sent_raw1, sent_label1)[0])

    ### Once...
    #data1_del, label1_del, n1_del = delete.do(filtering_clauses(sent_raw1, sent_label1))
    #data1_mov, label1_mov, n1_mov = move.do(filtering_clauses(sent_raw1, sent_label1))
    #print('*******************************************************')
    #print(data1_del)
    #print('*******************************************************')
    #print(data1_mov)

    total_new_data = total_new_data + raw_data #+ data1_del + data1_mov
    total_new_label = total_new_label + label_data #+ label1_del + label1_mov
    tot_new_num = tot_new_num + ori_num #+ n1_del + n1_mov


    n_cycle = 0
    while(True):

        n_cycle += 1
        
        ### Multiple...
        print(' ---> {} cycle start...'.format(n_cycle))
        
        data_rep, label_rep, n_rep = replace.do([sent_raw1, sent_label1])    
        #data_repENT, label_repENT, n_repENT = replace_ent.do([sent_raw2, sent_label2])
        data_ins, label_ins, n_ins = insert.do([sent_raw1, sent_label1])

        ###
        
        #data_rep_del, label_rep_del, n_rep_del = delete.do(filtering_clauses(data_rep, label_rep))
        #data_rep_mov, label_rep_mov, n_rep_mov = move.do(filtering_clauses(data_rep, label_rep))
        
        #data_repENT_del, label_repENT_del, n_repENT_del = delete.do(filtering_clauses(data_repENT, label_repENT))
        #data_repENT_mov, label_repENT_mov, n_repENT_mov = move.do(filtering_clauses(data_repENT, label_repENT))   

        #data_ins_del, label_ins_del, n_ins_del = delete.do(filtering_clauses(data_ins, label_ins))
        #data_ins_mov, label_ins_mov, n_ins_mov = move.do(filtering_clauses(data_ins, label_ins))

        total_new_data = total_new_data +  data_rep + data_ins#+  data_repENT + data_rep_del + data_rep_mov + data_repENT_del + data_repENT_mov + data_ins_del + data_ins_mov
        total_new_label = total_new_label +  label_rep + label_ins #+  label_repENT+ label_rep_del + label_rep_mov + label_repENT_del + label_repENT_mov + label_ins_del + label_ins_mov
        tot_new_num = tot_new_num +  n_rep + n_ins #+  n_repENT + n_rep_del + n_rep_mov + n_repENT_del + n_repENT_mov + n_ins_del + n_ins_mov
        
        
        # store
        path_write = 'gen_data/logic/logic_' + str(tot_new_num) + '(' + str(multiple_n) + ').txt'
        store_new_sent(path_write, total_new_data, total_new_label)

        # break condition
        if tot_new_num > multiple_n * ori_num:
            print('BREAK: total {} is generated (including original data)'.format(tot_new_num))
            break
        
      
        break
    
else:
    print("[arg error!] please add at least one data type ('ner' or 'doc')")
    exit()
    
    
    
    
    
    
    
    
    
    
    
    
    
    