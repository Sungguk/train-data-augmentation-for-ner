import spacy
import numpy as np
import random
import random as rn
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from data_handling_for_heuristic import *
import os

from nltk.parse.stanford import StanfordParser
from configparser import ConfigParser

### parsing parameters.ini
config = ConfigParser()
config.read('parameters.ini')
read_file_path = config.get('file-path', 'read_file_path')
filter_none_entity = config.getboolean('replace', 'filter_none_entity')
window_size_replace = config.getint('replace', 'window_size')
alpha = config.getfloat('replace', 'alpha')
sim_thr = config.getfloat('replace', 'sim_thr')
n_candidates_replace = config.getint('replace', 'n_candidates')
window_size_insert = config.getint('insert', 'window_size')
n_candidates_insert = config.getint('insert', 'n_candidates')

### load resources
stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()
nlp = spacy.load('en')
os.environ['STANFORD_PARSER'] = 'C:\\stanford-parser-full-2016-10-31\\stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:\\stanford-parser-full-2016-10-31\\stanford-parser-3.7.0-models.jar'
cons_parser = StanfordParser(model_path = 'edu\\stanford\\nlp\\models\\lexparser\\englishPCFG.ser.gz')

### load data
raw_data, label_data = load_conll2003(read_file_path)


class Test():
    def __init__(self):
        print((random.getstate()))


class Replace_ENT():

    def __init__(self):
        self.LOC_data = load('resources/entity_info/LOC_data(onlyCONLL)')
        self.PER_data = load('resources/entity_info/PER_data(onlyCONLL)')
        self.ORG_data = load('resources/entity_info/ORG_data(onlyCONLL)')
        print('> [Replace_ENT]: class is created')
    
    def extract_label(self, input_entity, type_ent):
        splited_entity = input_entity.split()
        temp = ''
        if len(splited_entity) == 1:
            label = 'B-' + type_ent
            return label
        else:
            total_label = ''
            label = 'I-' + type_ent
            for i in range(0, len(splited_entity)):
                if i==0:
                    total_label += 'B-' + type_ent
                else:
                    total_label += ' '
                    total_label += label
            return total_label     
        
    def replace_entity(self, input_entity, type_ent):
        if type_ent == 'LOC':
            gen_num = (rn.randrange(1, len(self.LOC_data)))
            new_entity = self.LOC_data[gen_num]
            new_label = self.extract_label(new_entity, type_ent)
        elif type_ent == 'PER':
            gen_num = (rn.randrange(1, len(self.PER_data)))
            new_entity = self.PER_data[gen_num]
            new_label = self.extract_label(new_entity, type_ent)
        elif type_ent == 'ORG':    
            gen_num = (rn.randrange(1, len(self.ORG_data)))
            new_entity = self.ORG_data[gen_num]
            new_label = self.extract_label(new_entity, type_ent)
        return new_entity.split(), new_label.split() # list형으로 변환.

    def is_only_MISC(self, label_list):
        tkn = False 
        if 'B-MISC' in label_list:
            if not 'B-PER' in label_list:
                if not 'B-ORG' in label_list:
                    if not 'B-LOC' in label_list:
                        tkn = True
        return tkn 

    def exist_ENTITY(self, label_list):
        tkn = False
        for token in label_list:
            if token != 'O': # 엔티티가 하나라도 있으면.
                tkn = True
        return tkn
        
    def do(self, data):
        raw_data = data[0]
        label_data = data[1]
        print('>>> [Replace_ENT Start!]: (number of input sent = {})'.format(len(raw_data)))
        new_sent_raw_list = []
        new_sent_label_list = []

        for i, sent in enumerate(raw_data):
            list_raw_data = raw_data[i].split()
            list_label_data = label_data[i].split()
            
            # Initialization
            temp_entity = ''
            start_ent_tkn = -1
            j = 0
            new_sent_raw = ''
            new_sent_label = ''

            if self.is_only_MISC(list_label_data) == True:
                continue # 문장에서 엔티티가 MISC만 있는 경우는 PASS
            
            if self.exist_ENTITY(list_label_data) == False:
                continue # 문장에서 엔티티가 아예 없는 경우는 PASS
            
            #print(list_raw_data)
            #print(list_label_data)    
            #print('------------------------------------')
            
            while j < len(list_label_data):
                type_ent = list_label_data[j].split('-')[-1]
                #print(j, list_label_data[j], list_raw_data[j])
                if list_label_data[j] != 'O' and list_label_data[j] != 'B-MISC' and list_label_data[j] != 'I-MISC':
                    if j == len(list_label_data)-1: # sent에서 마지막 token (따로 처리해줘야함.)
                        if list_label_data[j].split('-')[0] == 'B':
                            start_ent_tkn = j
                            ### 교체시작! (자기 혼자만.)
                            temp_entity += list_raw_data[j]
                            del list_raw_data[j]
                            del list_label_data[j]                   
                            
                            new_entity, new_label = self.replace_entity(temp_entity, type_ent)
                            new_sent_raw = list_raw_data[:j] + new_entity + list_raw_data[j:]
                            new_sent_label = list_label_data[:j] + new_label + list_label_data[j:]
                            
                            list_raw_data = new_sent_raw
                            list_label_data = new_sent_label                    
                            
                            j = j - 1 
                            j += len(new_entity)

                        elif list_label_data[j].split('-')[0] == 'I':

                            ### 교체시작! (history 합쳐서.)
                            temp_entity += ' '
                            temp_entity += list_raw_data[j]
                            del list_raw_data[j]
                            del list_label_data[j]                   
                            
                            #print(temp_entity)
                            new_entity, new_label = self.replace_entity(temp_entity, type_ent)
                            #print('new ->', new_entity, new_label)
                            new_sent_raw = list_raw_data[:j] + new_entity + list_raw_data[j:]
                            new_sent_label = list_label_data[:j] + new_label + list_label_data[j:]    
                            
                            list_raw_data = new_sent_raw
                            list_label_data = new_sent_label 
                            
                            j = j - 1 
                            j += len(new_entity) 

                    else: # All except for last one
                        if list_label_data[j].split('-')[0] == 'B':
                            start_ent_tkn = j
                            if list_label_data[j+1].split('-')[0] == 'I': 
                                ## 저장
                                temp_entity += list_raw_data[j]
                                del list_raw_data[j] # 현재 'B' 제거 (저장을 위해)
                                del list_label_data[j] # 현재 'B' 제거 (저장을 위해)                   
                                j = j - 1                       

                            else: # 'B' 또는 'O" 이라면..       
                                ### 교체시작! (자기 혼자만.)
                                temp_entity += list_raw_data[j]                        
                                del list_raw_data[j] # 현재 'B' 제거 (교체를 위해)
                                del list_label_data[j] # 현재 'B' 제거 (교체를 위해)         
                                new_entity, new_label = self.replace_entity(temp_entity, type_ent)
                                new_sent_raw = list_raw_data[:j] + new_entity + list_raw_data[j:]
                                new_sent_label = list_label_data[:j] + new_label + list_label_data[j:]
                                list_raw_data = new_sent_raw
                                list_label_data = new_sent_label   
                                j = j - 1 
                                j += len(new_entity)
                                temp_entity = '' # 초기화
                        
                        elif list_label_data[j].split('-')[0] == 'I':
                            if list_label_data[j+1].split('-')[0] == 'I':  
                                ## 저장
                                temp_entity += ' '
                                temp_entity += list_raw_data[j]                        
                                del list_raw_data[j]
                                del list_label_data[j]            
                                j = j - 1                         

                            else: # 'B' 또는 'O" 이라면..
                                ### 교체시작! (history 합쳐서.)
                                temp_entity += list_raw_data[j]
                                del list_raw_data[j]
                                del list_label_data[j]
                                                       
                                new_entity, new_label = self.replace_entity(temp_entity, type_ent)
                                new_sent_raw = list_raw_data[:j] + new_entity + list_raw_data[j:]
                                new_sent_label = list_label_data[:j] + new_label + list_label_data[j:]
                                
                                list_raw_data = new_sent_raw
                                list_label_data = new_sent_label
                                
                                j = j - 1 
                                j += len(new_entity)
                                temp_entity = '' # 초기화
                j += 1
            #print(list_raw_data)
            #print(list_label_data)
            
            # add new sentence
            new_sent_raw_list.append(' '.join(list_raw_data))
            new_sent_label_list.append(' '.join(list_label_data))
            #print('\n')
        print('>>> [Replace_ENT Done!]: (number of output (new) sent = {})'.format(len(new_sent_raw_list)))
        return new_sent_raw_list, new_sent_label_list, len(new_sent_raw_list) 
        
        
class Move():

    def __init__(self):
        self.sent_clauses = ['because', 'before', 'until', 'after', 'while', 'if', 'since', 'when', 'as', 'Because', 'Before', 'Until', 'After', 'While', 'If', 'Since', 'When', 'As']
        print('> [Move]: class is created')

    def del_ent_in_list(self, delete_list, X):
        # token 단위로 확인하면서 삭제한다
        # call to ( boycott ) 이면 (과 )는 삭제된다. token단위이므로... (token단위 기준: space)
        # X: ['EU rejects German call to boycott British lamb .', ... ]
        # Y: ['I-ORG O I-MISC O O O I-MISC O O', ... ]
        # delete_list = [')','(', ';', ...]
        i = 0
        while i < len(X):
            if any(str(X[i]) in t for t in delete_list):
                #del Y[i]
                del X[i]
            #else:
            i+=1               
        return X #, Y

    def del_ent_in_string(self, delete_list, X):
    # char 단위로 확인하면서 삭제한다
    # 예를 들어, boycott이 아니라 boy(cott이면 해당 token은 삭제된다. 이처럼 char레벨까지 체크한다.
    # X: ['EU rejects German call to boycott British lamb .', ... ]
    # Y: ['I-ORG O I-MISC O O O I-MISC O O', ... ]
    # delete_list = [')','(', ';', ...]    
        i=0
        while i < len(X):
            #print(str(sp_test[i]))
            list_X = list(X[i])
            if len(list_X)==1:
                break
            for char in list_X:
                if any(char in t for t in delete_list):
                    #print(X)
                    #print(Y)
                    #del Y[i]
                    del X[i]
                    break
            #else:
            i+=1
        return X#, Y        
        
    def traverseTree(self, tree, temp_sentence):
    #     if tree.label() == 'ROOT': # 처음에 initialization
    #         index_SBAR = -1
    #         list_SBAR = [0] * len(sent_list)
        #print("tree:", tree)
        #print("======> tree.pos()", tree.pos())
        #print("======> tree.height()", tree.height())
        #print("======> tree.label()", tree.label())
        #print("======> tree.leaves()", tree.leaves())
        #print("======> current_depth", current_depth)
        #print('======> current depth list', list_current_depth)    
        #print('**************************************************')
        #print('\n')
        if type(tree) == nltk.tree.Tree:
            # Initialication
            current_height = tree.height()
            ### 로직 한계점
            ### 만약 SBAR안에 SBAR이 있는 경우 구분하지 못한다. 또는 SBAR이 또다른 SBAR과 바로 옆에 붙어있는 경우 구분하지 못한다.
            ### 그러면 나중에 제일 첫 글자만 참고해서 어떤 부사절인지 보는데, 만약 2개 이상 SBAR이 포함되어있거나 붙어있는 경우
            ### index가 제일 앞에 있는 SBAR만 참고하게 된다. (예를 들어, because SBAR이 뒤에 있는 경우 무시하게 된다)
            # SBAR
            if tree.label() == 'SBAR':
                #print('SBAR start!!!')
                if len(tree.leaves()) <= 1: # SBAR token개수가 1개 이하이면 return. 
                    return None
                for i, token in enumerate(temp_sentence):
                    #print('temp_sentence[i]:', temp_sentence[i], ', tree.leaves()[0]:',tree.leaves()[0]) 
                    #print('temp_sentence[i+1]:', temp_sentence[i+1], ', tree.leaves()[1]:', tree.leaves()[1])
                    #if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+1] == tree.leaves()[1]:
                        #print('break!!!!!!!!!!!!!!!!!!!!!!')
                        self.index_SBAR = i
                        break
                if not self.index_SBAR == -1:
                    for j in range(self.index_SBAR, self.index_SBAR+len(tree.leaves())):
                        self.list_SBAR[j] = 1   
                return None
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                self.traverseTree(subtree, temp_sentence) # recursive

    def re_numbering(self, list_SBAR):
    # ex. [1,1,1,0,0,1,1,0,1,1] => [1,1,1,0,0,2,2,0,3,3]
    # 후처리하기 쉽게하기 위해 numbering을 다시한다.
        cur_idx = 1
        new_list_SBAR = [0] * len(list_SBAR)
        for i, _ in enumerate(list_SBAR):
            if list_SBAR[i]!=0:
                new_list_SBAR[i] = cur_idx
            if not i==len(list_SBAR)-1:
                if list_SBAR[i]!=0 and list_SBAR[i+1]==0:
                    cur_idx += 1
        return new_list_SBAR
    
    def what_clause(self, renumbered_list_SBAR, token_raw_list):
        clause_list = ['none'] * len(renumbered_list_SBAR)

        for i, num in enumerate(renumbered_list_SBAR):
            if i==0 and renumbered_list_SBAR[i]!=0:
                if token_raw_list[i] in self.sent_clauses:
                    clause_list[i] = token_raw_list[i]
            if renumbered_list_SBAR[i-1]==0 and renumbered_list_SBAR[i]!=0:
                if token_raw_list[i] in self.sent_clauses:
                    #print(token_raw_list[i])
                    clause_list[i] = token_raw_list[i]    
        temp_index = -1
        for i, num in enumerate(renumbered_list_SBAR):
            if clause_list[i]!='none':
                temp_clause = clause_list[i]
                temp_index = renumbered_list_SBAR[i]
            if renumbered_list_SBAR[i] == temp_index:
                clause_list[i] = temp_clause         
        return clause_list
    
    def move_with_SBAR(self, list_SBAR, token_raw_list, token_label_list, clause_list):
        cnt_headmove = 0
        cnt_tailmove = 0
        # 주의: list_SBAR, sent_raw_list, sent_label_list의 길이는 모두 같다.

        new_raw_sent_list = [] # 문장 여러개
        new_label_sent_list = []
        #print(token_raw_list)
        #print(list_SBAR)
        #print('------------------------------------------')    

        for i, _ in enumerate(list_SBAR):
            ### 문장 머리에 있을 경우
            #######################
            if i==0 and list_SBAR[i] != 0 and clause_list[i] in self.sent_clauses: # 문장 머리에 sent clause SBAR이 있는 경우..
                sbar_num = list_SBAR[i]
                temp = []
                for i, x in enumerate(list_SBAR):
                    if x == sbar_num:
                        temp.append(token_raw_list[i])
                        end_pos = i+1
                deleted_sent = token_raw_list[:end_pos]
                remained_sent = token_raw_list[end_pos:]
                deleted_label = token_label_list[:end_pos]
                remained_label = token_label_list[end_pos:]
                #print(remained_sent + deleted_sent)
                #print(remained_label + deleted_label)
                cnt_headmove += 1
                temp_sent = remained_sent + deleted_sent
                temp_label = remained_label + deleted_label
                new_raw_sent_list.append(' '.join(temp_sent))
                new_label_sent_list.append(' '.join(temp_label))
            ### 문장 꼬리에 있을 경우
            #######################
            if i==len(list_SBAR)-2 and list_SBAR[i] !=0 and clause_list[i] in self.sent_clauses: # 문장 꼬리에 sent clause SBAR이 있는 경우..
                sbar_num = list_SBAR[i]
                temp = []
                tkn = False
                for i, x in enumerate(list_SBAR):
                    if x == sbar_num:
                        temp.append(token_raw_list[i])

                        if tkn == False:
                            start_pos = i
                            tkn = True
                remained_sent = token_raw_list[:start_pos]
                deleted_sent = token_raw_list[start_pos:]
                remained_label = token_label_list[:start_pos]
                deleted_label = token_label_list[start_pos:]
                #print(deleted_sent + remained_sent)
                #print(deleted_label + remained_label)               
                cnt_tailmove += 1
                temp_sent = deleted_sent + remained_sent
                temp_label = deleted_label + remained_label
                new_raw_sent_list.append(' '.join(temp_sent))
                new_label_sent_list.append(' '.join(temp_label))
        return new_raw_sent_list, new_label_sent_list

    def do(self, data):
        sent_raw = data[0]
        sent_label = data[1]
        print('>>> [Move Start!]: (number of input sent = {})'.format(len(sent_raw)))
        new_sent_list = []
        new_label_list = [] 
        for i, _ in enumerate(sent_raw):
        #     print('\n')
        #     print(sent_raw[i])
            token_raw_list = sent_raw[i].split()
            token_label_list = sent_label[i].split()
            #         
            if '(' in list(sent_raw[i]):
                continue
            if ')' in list(sent_raw[i]):
                continue
            ### pos tagging
            pos_sent_raw_list = nltk.pos_tag(token_raw_list)
            #pos_sent_raw_list = nltk.pos_tag(token_raw_list)
            ### parsing
            parse_result = cons_parser.tagged_parse(pos_sent_raw_list)
            ###################################
            """ SBAR 이동 """
            ###################################
            # initialization 
            self.index_SBAR = -1
            self.list_SBAR = [0] * len(token_raw_list)    
            for tree in parse_result:
                parse_tree = tree
                self.traverseTree(parse_tree, token_raw_list)
      
            ##############################################
            """ Boundary 검사: 엔티티가 쪼개져 있으면 수정 """    
            ##############################################
            # parsing 결과를 보정
            tkn = True
            while(tkn):
                btn = False
                for i, num in enumerate(self.list_SBAR):
                    if i!=len(self.list_SBAR)-1:
                        if self.list_SBAR[i] == 1 and self.list_SBAR[i+1] == 0:
                            btn = True
                if btn == False and self.list_SBAR[-1] == 1:
                    tkn = False
                if not 1 in self.list_SBAR:
                    tkn = False
                for i, num in enumerate(self.list_SBAR):
                    if i!=0: # 첫 번째 index는 제외
                        if self.list_SBAR[i-1] == 1 and self.list_SBAR[i] == 0: # find SBAR boundary!
                            if token_label_list[i-1] != 'O': # 엔티티라면
                                if token_label_list[i].split('-')[0] == 'I': # (좀 더 구체적으로..) INSIDE 계열의 엔티티라면..
                                    self.list_SBAR[i] = 1
                                else: # BEGIN 계열이면 그냥 pass
                                    tkn = False # break
                            else: # 엔티티가 아니라면
                                tkn = False # break             
        #     ### for printf
        #     print(list_SBAR)
        #     if 1 in list_SBAR:
        #         temp = ''
        #         for k, tkn in enumerate(token_raw_list):
        #             if list_SBAR[k] == 1:
        #                 temp = temp+token_raw_list[k]+' '
        #         print(temp)
            # generation with SBAR
            if 1 in self.list_SBAR: # 현재 문장에서 SBAR이 있으면...
                # renumbering
                self.list_SBAR = self.re_numbering(self.list_SBAR)
                clause_list = self.what_clause(self.list_SBAR, token_raw_list)
        #         print(len(clause_list), len(list_SBAR))
        #         print(clause_list)
        #         print(list_SBAR)
        #         temp = ''
        #         for k, tkn in enumerate(token_raw_list):
        #             if list_SBAR[k] == 1:
        #                 temp = temp+token_raw_list[k]+' '
        #         print(temp)        
                new_sent_raw_list, new_sent_label_list = self.move_with_SBAR(self.list_SBAR, token_raw_list, token_label_list, clause_list)
                if len(new_sent_raw_list) != 0: # 새로운 문장이 생성되었는가?
                    # 주의: append가 아니라 +로해야됨. new_sent_list가 list이기 때문에
                    new_sent_list += new_sent_raw_list
                    new_label_list += new_sent_label_list
        #             ### for printf 
        #             print('***********************************')
        #             print(sent_raw[i])            
                    
        #             ### for printf 
        #             print(list_SBAR)
        #             if 1 in list_SBAR:
        #                 temp = ''
        #                 for k, tkn in enumerate(token_raw_list):
        #                     if list_SBAR[k] == 1:
        #                         temp = temp+token_raw_list[k]+' '
        #                 print(temp)            
            # 엔티티 포함되어 있으면 냅둔다.
        #     for i, _ in enumerate(list_SBAR):
        #         if not i==0:
        #             if list_SBAR[i-1]==0 and list_SBAR[i]==1:
        #         else: # i==0일 때...
        #         print('------------------------------------------------')            
        #         print(sent_raw[i])
        #         print('------------------------------------------------')
        #         print(temp)
        #         print(list_SBAR)
        #         print('\n')
        #         print('------------------------------------------------')
        print('>>> [Move Done!]: (number of output (new) sent = {})'.format(len(new_sent_list))) 
        return new_sent_list, new_label_list, len(new_sent_list)


class Delete():

    def __init__(self):
        self.sent_clauses = ['because', 'before', 'until', 'after', 'while', 'if', 'since', 'when', 'as', 'where', 'Because', 'Before', 'Until', 'After', 'While', 'If', 'Since', 'When', 'As']
        self.noun_clauses = ['which', 'who', 'to']
        # to는 to부정사를 지칭하는 거임. 좀 더 정확하게 하려면 다음에 동사인 것을 확인해야 하지만, SBAR인점과 첫 글자가 to이라는 것은 to부정사 확률이 높음.
        # 명사절에서 that절은 빼자. that이 워낙 경우의 수가 다양해서.. 일단은 배제하자.
        print('> [Delete]: class is created')

    def del_ent_in_list(self, delete_list, X):
        # token 단위로 확인하면서 삭제한다
        # call to ( boycott ) 이면 (과 )는 삭제된다. token단위이므로... (token단위 기준: space)
        # X: ['EU rejects German call to boycott British lamb .', ... ]
        # Y: ['I-ORG O I-MISC O O O I-MISC O O', ... ]
        # delete_list = [')','(', ';', ...]
        i = 0
        while i < len(X):
            if any(str(X[i]) in t for t in delete_list):
                #del Y[i]
                del X[i]
            #else:
            i+=1               
        return X #, Y

    def del_ent_in_string(self, delete_list, X):
    # char 단위로 확인하면서 삭제한다
    # 예를 들어, boycott이 아니라 boy(cott이면 해당 token은 삭제된다. 이처럼 char레벨까지 체크한다.
    # X: ['EU rejects German call to boycott British lamb .', ... ]
    # Y: ['I-ORG O I-MISC O O O I-MISC O O', ... ]
    # delete_list = [')','(', ';', ...]    
        i=0
        while i < len(X):
            #print(str(sp_test[i]))
            list_X = list(X[i])
            if len(list_X)==1:
                break
            for char in list_X:
                if any(char in t for t in delete_list):
                    #print(X)
                    #print(Y)
                    #del Y[i]
                    del X[i]
                    break
            #else:
            i+=1
        return X#, Y        
        
    def traverseTree(self, tree, temp_sentence):

    #     if tree.label() == 'ROOT': # 처음에 initialization
    #         index_SBAR = -1
    #         list_SBAR = [0] * len(sent_list)
        #print("tree:", tree)
        #print("======> tree.pos()", tree.pos())
        #print("======> tree.height()", tree.height())
        #print("======> tree.label()", tree.label())
        #print("======> tree.leaves()", tree.leaves())
        #print("======> current_depth", current_depth)
        #print('======> current depth list', list_current_depth)    
        #print('**************************************************')
        #print('\n')

        if type(tree) == nltk.tree.Tree:
            # Initialication
            current_height = tree.height()
            
            ### 로직 한계점
            ### 만약 SBAR안에 SBAR이 있는 경우 구분하지 못한다. 또는 SBAR이 또다른 SBAR과 바로 옆에 붙어있는 경우 구분하지 못한다.
            ### 그러면 나중에 제일 첫 글자만 참고해서 어떤 부사절인지 보는데, 만약 2개 이상 SBAR이 포함되어있거나 붙어있는 경우
            ### index가 제일 앞에 있는 SBAR만 참고하게 된다. (예를 들어, because SBAR이 뒤에 있는 경우 무시하게 된다)
            
            # SBAR
            if tree.label() == 'SBAR':
                #print('SBAR start!!!')
                if len(tree.leaves()) <= 1: # SBAR token개수가 1개 이하이면 return. 
                    return None
                for i, token in enumerate(temp_sentence):
                    #print('temp_sentence[i]:', temp_sentence[i], ', tree.leaves()[0]:',tree.leaves()[0]) 
                    #print('temp_sentence[i+1]:', temp_sentence[i+1], ', tree.leaves()[1]:', tree.leaves()[1])
                    #if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+len(tree.leaves())-1] == tree.leaves()[-1]:
                    if temp_sentence[i] == tree.leaves()[0] and temp_sentence[i+1] == tree.leaves()[1]:
                        #print('break!!!!!!!!!!!!!!!!!!!!!!')
                        self.index_SBAR = i
                        break
                if not self.index_SBAR == -1:
                    for j in range(self.index_SBAR, self.index_SBAR+len(tree.leaves())):
                        self.list_SBAR[j] = 1   
                return None

        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                self.traverseTree(subtree, temp_sentence) # recursive

    def re_numbering(self, list_SBAR):
    # ex. [1,1,1,0,0,1,1,0,1,1] => [1,1,1,0,0,2,2,0,3,3]
    # 후처리하기 쉽게하기 위해 numbering을 다시한다.
        cur_idx = 1
        new_list_SBAR = [0] * len(list_SBAR)
        for i, _ in enumerate(list_SBAR):
            if list_SBAR[i]!=0:
                new_list_SBAR[i] = cur_idx
            if not i==len(list_SBAR)-1:
                if list_SBAR[i]!=0 and list_SBAR[i+1]==0:
                    cur_idx += 1
        return new_list_SBAR

    def gen_with_SBAR(self, list_SBAR, token_raw_list, token_label_list):
        
        # 주의: list_SBAR, sent_raw_list, sent_label_list의 길이는 모두 같다.
        new_raw_sent_list = [] # 문장 여러개 
        new_label_sent_list = []
        
        #print(token_raw_list)
        #print(list_SBAR)
        #print('------------------------------------------')    

        for i, _ in enumerate(list_SBAR):
            temp_raw_str = ''
            temp_label_str = ''

            ##########################
            ###### @@ 로직 똑같음 ######
            ##########################
            if i==0 and list_SBAR[i] != 0: # 앞 boundary일 경우의 수 1
            ##########################    
            ##########################    
                if str(token_raw_list[i]) in self.sent_clauses: # 삭제, 분리
                    
                    # 엔티티가 해당 clause에 있는지 확인
                    is_ent_in_clause = False
                    for k, _ in enumerate(list_SBAR):
                        if list_SBAR[k]==list_SBAR[i]: # 해당 번호
                            if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                is_ent_in_clause = True
                                
                    if is_ent_in_clause == False: # 엔티티가 없으면.. 해당 clause 삭제
                        for m, n in enumerate(list_SBAR):
                            if n != list_SBAR[i]: # clause를 삭제하자. 즉, clause에 속하지 않는 것들만 추리자.
                                temp_raw_str += str(token_raw_list[m])
                                temp_raw_str += ' '
                                temp_label_str += str(token_label_list[m])
                                temp_label_str += ' '
                                
                        ### clause가 삭제된 문장 저장
                        #print(temp_raw_str)
                        self.cnt_MainClause_delete_SBARwithoutENT += 1
                        new_raw_sent_list.append(temp_raw_str)
                        new_label_sent_list.append(temp_label_str)
                        
                    if is_ent_in_clause == True: # 엔티티가 있으면..
                        fir_tkn = False
                        for m, n in enumerate(list_SBAR):
                            if n == list_SBAR[i]: # clause에 속하는 것들만 추리자
                                if fir_tkn == False: # 첫 번째 글자는 pass하자. (ex. because, if, while 등등)
                                    fir_tkn = True
                                    continue
                                temp_raw_str += str(token_raw_list[m])
                                temp_raw_str += ' '
                                temp_label_str += str(token_label_list[m])
                                temp_label_str += ' '
                                
                        ### clause가 삭제된 문장 저장
                        #print(temp_raw_str)
                        self.cnt_AdverbialClause_reuse_SBAR += 1
                        new_raw_sent_list.append(temp_raw_str)
                        new_label_sent_list.append(temp_label_str)
                        
                        # 엔티티가 clause외에도 있는지 확인
                        is_ent_out_clause = False
                        for k, _ in enumerate(list_SBAR):
                            if list_SBAR[k]!=list_SBAR[i]: # 해당 번호
                                if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                    is_ent_out_clause = True                
                        
                        # 엔티티가 clause외에 있으면 그것도 데이터확장에 사용.
                        temp_raw_str = ''
                        temp_label_str = ''
                        if is_ent_out_clause == True:
                            for m, n in enumerate(list_SBAR):
                                if n != list_SBAR[i]: # clause에 속하지 않는 것들만 추리자.
                                    temp_raw_str += str(token_raw_list[m])
                                    temp_raw_str += ' '
                                    temp_label_str += str(token_label_list[m])
                                    temp_label_str += ' '

                            ### clause가 삭제된 문장 저장
                            #print(temp_raw_str)
                            self.cnt_MainClause_delete_SBARwithENT += 1
                            new_raw_sent_list.append(temp_raw_str)
                            new_label_sent_list.append(temp_label_str)                

                elif str(token_raw_list[i]) in self.noun_clauses: # 삭제만함.

                    # 엔티티가 해당 clause에 있는지 확인
                    is_ent_in_clause = False
                    for k, _ in enumerate(list_SBAR):
                        if list_SBAR[k]==list_SBAR[i]: # 해당 번호
                            if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                is_ent_in_clause = True
                                
                    if is_ent_in_clause == False: # 엔티티가 없으면.. 해당 clause 삭제
                        for m, n in enumerate(list_SBAR):
                            if n != list_SBAR[i]: # clause를 삭제하자. 즉, clause에 속하지 않는 것들만 추리자.
                                temp_raw_str += str(token_raw_list[m])
                                temp_raw_str += ' '
                                temp_label_str += str(token_label_list[m])
                                temp_label_str += ' '
                                
                        ### clause가 삭제된 문장 저장
                        #print(temp_raw_str)
                        self.cnt_MainClause_delete_SBARwithoutENT += 1
                        new_raw_sent_list.append(temp_raw_str)
                        new_label_sent_list.append(temp_label_str)                

            ##########################                
            ###### @@ 로직 똑같음 ######
            ##########################            
            if i != 0:
                if list_SBAR[i-1]==0 and list_SBAR[i] != 0: # 앞 boundary일 경우의 수 2
            ##########################
            ##########################
            
                    if str(token_raw_list[i]) in self.sent_clauses: # 삭제, 분리

                        # 엔티티가 해당 clause에 있는지 확인
                        is_ent_in_clause = False
                        for k, _ in enumerate(list_SBAR):
                            if list_SBAR[k]==list_SBAR[i]: # 해당 번호
                                if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                    is_ent_in_clause = True

                        if is_ent_in_clause == False: # 엔티티가 없으면.. 해당 clause 삭제
                            for m, n in enumerate(list_SBAR):
                                if n != list_SBAR[i]: # clause를 삭제하자. 즉, clause에 속하지 않는 것들만 추리자.
                                    temp_raw_str += str(token_raw_list[m])
                                    temp_raw_str += ' '
                                    temp_label_str += str(token_label_list[m])
                                    temp_label_str += ' '

                            ### clause가 삭제된 문장 저장
                            #print(temp_raw_str)
                            self.cnt_MainClause_delete_SBARwithoutENT += 1
                            new_raw_sent_list.append(temp_raw_str)
                            new_label_sent_list.append(temp_label_str)

                        if is_ent_in_clause == True: # 엔티티가 있으면..
                            fir_tkn = False
                            for m, n in enumerate(list_SBAR):
                                if n == list_SBAR[i]: # clause에 속하는 것들만 추리자
                                    if fir_tkn == False: # 첫 번째 글자는 pass하자. (ex. because, if, while 등등)
                                        fir_tkn = True
                                        continue
                                    temp_raw_str += str(token_raw_list[m])
                                    temp_raw_str += ' '
                                    temp_label_str += str(token_label_list[m])
                                    temp_label_str += ' '

                            ### clause가 삭제된 문장 저장
                            #print(temp_raw_str)
                            self.cnt_AdverbialClause_reuse_SBAR += 1
                            new_raw_sent_list.append(temp_raw_str)
                            new_label_sent_list.append(temp_label_str)

                            # 엔티티가 clause외에도 있는지 확인
                            is_ent_out_clause = False
                            for k, _ in enumerate(list_SBAR):
                                if list_SBAR[k]!=list_SBAR[i]: # 해당 번호
                                    if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                        is_ent_out_clause = True                

                            # 엔티티가 clause외에 있으면 그것도 데이터확장에 사용.
                            temp_raw_str = ''
                            temp_label_str = ''
                            if is_ent_out_clause == True:
                                for m, n in enumerate(list_SBAR):
                                    if n != list_SBAR[i]: # clause에 속하지 않는 것들만 추리자.
                                        temp_raw_str += str(token_raw_list[m])
                                        temp_raw_str += ' '
                                        temp_label_str += str(token_label_list[m])
                                        temp_label_str += ' '

                                ### clause가 삭제된 문장 저장
                                #print(temp_raw_str)
                                self.cnt_MainClause_delete_SBARwithENT += 1
                                new_raw_sent_list.append(temp_raw_str)
                                new_label_sent_list.append(temp_label_str)                

                    elif str(token_raw_list[i]) in self.noun_clauses: # 삭제만함.

                        # 엔티티가 해당 clause에 있는지 확인
                        is_ent_in_clause = False
                        for k, _ in enumerate(list_SBAR):
                            if list_SBAR[k]==list_SBAR[i]: # 해당 번호
                                if token_label_list[k]!='O': # 엔티티가 있다는 뜻
                                    is_ent_in_clause = True

                        if is_ent_in_clause == False: # 엔티티가 없으면.. 해당 clause 삭제
                            for m, n in enumerate(list_SBAR):
                                if n != list_SBAR[i]: # clause를 삭제하자. 즉, clause에 속하지 않는 것들만 추리자.
                                    temp_raw_str += str(token_raw_list[m])
                                    temp_raw_str += ' '
                                    temp_label_str += str(token_label_list[m])
                                    temp_label_str += ' '

                            # clause가 삭제된 문장 저장
                            #print(temp_raw_str)
                            self.cnt_MainClause_delete_SBARwithoutENT += 1
                            new_raw_sent_list.append(temp_raw_str)
                            new_label_sent_list.append(temp_label_str)                

        return new_raw_sent_list, new_label_sent_list        
        
        
    def do(self, data):
        sent_raw = data[0]
        sent_label = data[1]
        
        print('>>> [Delete Start!]: (number of input sent = {})'.format(len(sent_raw)))
        self.cnt_MainClause_delete_SBARwithoutENT = 0
        self.cnt_AdverbialClause_reuse_SBAR = 0
        self.cnt_MainClause_delete_SBARwithENT = 0

        new_sent_list = []
        new_label_list = [] 

        for i, _ in enumerate(sent_raw):
            
        #     print('***********************************')
        #     print(sent_raw[i])
            
            token_raw_list = sent_raw[i].split()
            #token_raw_list = test
            token_label_list = sent_label[i].split()
            #print(test)
            #print(sent_raw[i])
            #print(sent_label[i])
            
            if '(' in list(sent_raw[i]):
                continue
            if ')' in list(sent_raw[i]):
                continue
            

            ### pos tagging
            pos_sent_raw_list = nltk.pos_tag(token_raw_list)
            #pos_sent_raw_list = nltk.pos_tag(token_raw_list)
            ### parsing
            parse_result = cons_parser.tagged_parse(pos_sent_raw_list)

            # """ SBAR, 부사절 삭제 및 분리 """
            # initialization
            self.index_SBAR = -1
            self.list_SBAR = [0] * len(token_raw_list)    
            
            for tree in parse_result:
                parse_tree = tree
                self.traverseTree(parse_tree, token_raw_list)# tree traverse하면서 SBAR tree label을 찾음.
            
            #print(list_SBAR)
                                
            # """ Boundary 검사: 엔티티가 쪼개져 있으면 수정 """    
            # parsing 결과를 보정
            tkn = True
            while(tkn):
                btn = False
                for i, num in enumerate(self.list_SBAR):
                    if i!=len(self.list_SBAR)-1:
                        if self.list_SBAR[i] == 1 and self.list_SBAR[i+1] == 0:
                            btn = True
                if btn == False and self.list_SBAR[-1] == 1:
                    tkn = False
                if not 1 in self.list_SBAR:
                    tkn = False
                for i, num in enumerate(self.list_SBAR):
                    if i!=0: # 첫 번째 index는 제외
                        if self.list_SBAR[i-1] == 1 and self.list_SBAR[i] == 0: # find SBAR boundary!
                            if token_label_list[i-1] != 'O': # 엔티티라면
                                if token_label_list[i].split('-')[0] == 'I': # (좀 더 구체적으로..) INSIDE 계열의 엔티티라면..
                                    self.list_SBAR[i] = 1
                                else: # BEGIN 계열이면 그냥 pass
                                    tkn = False # break
                            else: # 엔티티가 아니라면
                                tkn = False # break            
            ### for printf 
        #     print(list_SBAR)
        #     if 1 in list_SBAR:
        #         temp = ''
        #         for k, tkn in enumerate(token_raw_list):
        #             if list_SBAR[k] == 1:
        #                 temp = temp+token_raw_list[k]+' '
        #         print(temp)

            # generation with SBAR           
            if 1 in self.list_SBAR: # 현재 문장에서 SBAR이 있으면...
                # renumbering
                self.list_SBAR = self.re_numbering(self.list_SBAR)
                new_sent_raw_list, new_sent_label_list = self.gen_with_SBAR(self.list_SBAR, token_raw_list, token_label_list)
                #print(new_sent_raw_list)
                #print('\n')
                if len(new_sent_raw_list) != 0: # 새로운 문장이 생성되었는가?
                    # 주의: append가 아니라 +로해야됨. new_sent_list가 list이기 때문에
                    new_sent_list += new_sent_raw_list
                    new_label_list += new_sent_label_list
                    
            # 엔티티 포함되어 있으면 냅둔다.
        #     for i, _ in enumerate(list_SBAR):

        #         if not i==0:
        #             if list_SBAR[i-1]==0 and list_SBAR[i]==1:

        #         else: # i==0일 때...
                #print('------------------------------------------------')            
                #print(sent_raw[i])
                #print('------------------------------------------------')
                #print(temp)
                #print(list_SBAR)
                #print('------------------------------------------------')
            #print('\n')
            
            #break
        print('>>> [Delete Done!]: (number of output (new) sent = {})'.format(len(new_sent_list))) 
        return new_sent_list, new_label_list, len(new_sent_list)
        
        
class Insert():

    def __init__(self):
        if filter_none_entity == True:
            self.sent_raw, self.sent_label = filtering_none_entity(raw_data, label_data)
        else:
            self.sent_raw = raw_data[:]
            self.sent_label = label_data[:] 
        self.friend_list = load('resources/for_insert/friend_list')
        self.voca = load('resources/for_insert/voca')
        print('> [Insert]: class is created')
    
    def similarity(self, me, other):
        me = nlp.vocab[me]
        other = nlp.vocab[other]
        if me.vector_norm == 0 or other.vector_norm == 0:
            return 0.0
        return np.dot(me.vector, other.vector) / (me.vector_norm * other.vector_norm)    
    
    def max_friend(self, friend_list, token):
        sim_score = [0] * len(friend_list)
        for i, each in enumerate(friend_list):
            sim_score[i] = self.similarity(token, friend_list[i])
        sim_score.sort(key=lambda x: x, reverse=True)
        sim_score = sim_score[:n_candidates_insert]
        sim_score = list(filter(lambda x: x!= 0.0, sim_score))
        if len(sim_score)==0:
            return 'none'
        #print(sim_score)
        #print('\n')
        #print(np.array(sim_score).max())
        choiced_score = random.choice(sim_score)
        #print(choiced_score)
        #print(np.array(sim_score).max(), '--', choiced_score)
        max1_index = np.where(sim_score == np.array(sim_score).max())[0][0]
        max_index = np.where(sim_score == choiced_score)[0][0]
        #print(max1_index, max_index)
        #print('\n')
        return friend_list[max_index]
          
    def insert_module(self, token, pos):
        #print('------>',token, pos)
        try:voca_index = self.voca.index((token, pos))
        except(ValueError):
            #print('*********>', 'none')
            return 'none'
        if len(self.friend_list[voca_index])==0:
            #print('*********>', 'none')
            return 'none'
        else:
            new_token = self.max_friend(self.friend_list[voca_index], token)
            #print('*********>',new_token)
            return new_token        

    def window_filtering(self, sent_label, window_size=3):
        list_sent_label = sent_label.split()
        new_list = [0] * len(list_sent_label)
        ### 순방향
        for i, _ in enumerate(list_sent_label):
            temp = list_sent_label[i:i+1+window_size]
            #print(temp)
            tkn = False
            for each in temp:
                if each != 'O':
                    tkn = True
            if tkn == True:
                new_list[i] = 1
        ### 역방향
        for i  in range(len(list_sent_label)-1, -1, -1):
            idx = i-window_size
            if idx < 0:
                idx = 0
            temp = list_sent_label[idx:i]
            tkn = False
            for each in temp:
                if each != 'O':
                    tkn = True
            if tkn == True:
                new_list[i] = 1      
        return new_list  

    def do(self, data):
        sent_raw = data[0]
        sent_label = data[1]
        print('>>> [Insert Start!]: (number of input sent = {})'.format(len(sent_raw)))
        insert_sent_cnt = 0
        new_sent_list = []
        new_label_list = [] 
        for i, _ in enumerate(sent_raw):
            #list_sent_raw = sent_raw[i].split()
            str_sent_raw = preprocessing_for_spacy(sent_raw[i])
            list_sent_nlp = nlp(str_sent_raw)
            list_sent_raw = str_sent_raw.split()
            list_sent_label = sent_label[i].split()
            #print(str_sent_raw)
            ### 조건에 맞는 token들을 불러온다
            pre_token_pos = 'none'
            insert_on = False
            add_idx = 0
            ### filtering with window size based on entities
            filtered_list = self.window_filtering(sent_label[i], window_size_insert)
            #print(sent_raw[i])
            #print(sent_label[i])
            if len(list_sent_nlp) != len(list_sent_label):
                #print('[ERROR]: len(list_sent_nlp) != len(list_sent_label)')
                continue
            for i, token in enumerate(list_sent_nlp):
                new_token = 'none' # 계속 'none'으로 남아있다면 추가하지 않을 것임.
                # 엔티티는 건드리지 말자.
                if list_sent_label[i+add_idx] != 'O':
                    continue
                if filtered_list[i+add_idx] == 0: # filtering with window size based on entities
                    continue
                #print('>>>', token, token.pos_, pre_token_pos)
                #################
                ### INSERT_MODULE  
                str_token = str(token).lower()
                if token.pos_ == 'NOUN' or token.pos_ == 'PROPN': # 오로지 하나의 명사일경우...
                    if i==0:
                        str_token = lemmatiser.lemmatize(str_token, 'n')
                        new_token = self.insert_module(str_token, 'noun')
                    else:
                        if pre_token_pos != 'ADJ':
                            if pre_token_pos != 'NOUN':
                                if pre_token_pos != 'PROPN':
                                    str_token = lemmatiser.lemmatize(str_token, 'n')
                                    new_token = self.insert_module(str_token, 'noun')
                elif token.pos_ == 'VERB': 
                    str_token = lemmatiser.lemmatize(str_token, 'v')
                    new_token = self.insert_module(str_token, 'verb')
                elif token.pos_ == 'ADJ':
                    new_token = self.insert_module(str_token, 'adj')
                
                ################
                ### REAL INSERT
                pre_token_pos = token.pos_
                if new_token == 'none': # insert된 것이 하나도 없으면 pass.
                    continue # pass
                else:
                    insert_on = True # insert 한 번 이상 되었음.
                    list_sent_raw.insert(i+add_idx, new_token)
                    list_sent_label.insert(i+add_idx, 'O')
                    filtered_list.insert(i+add_idx, -1)
                    add_idx += 1
                #print('\n')
            #print(' '.join(list_sent_raw))
            #print(' '.join(list_sent_label))
            #print('\n')
            ### Append new sent list
            if insert_on == True:
                insert_sent_cnt += 1
                new_sent_list.append(' '.join(list_sent_raw))
                new_label_list.append(' '.join(list_sent_label))
            #break
        print('>>> [Insert Done!]: (number of output (new) sent = {})'.format(insert_sent_cnt))            
        return new_sent_list, new_label_list, insert_sent_cnt
    
    
class Replace():

    def __init__(self):
        print('> [Replace]: class is created')
        
    def wordnet(self, word, pos):
        if type(word) != 'str':
            word = str(word)   
        synsets = wn.synsets(word, pos)
        list_synsets = []
        for i, syn in enumerate(synsets):
            #print('%d. %s' % (i, syn.name()))
            li = (syn.lemma_names())
            list_synsets += (li)
            for hyper_syn in syn.hypernyms():
                list_synsets += (hyper_syn.lemma_names())
            for hypo_syn in syn.hyponyms():
                list_synsets += (hypo_syn.lemma_names())
            for holo_syn in syn.part_holonyms():
                list_synsets += (holo_syn.lemma_names())
            for mero_syn in syn.part_meronyms():
                list_synsets += (mero_syn.lemma_names())

        # 똑같은 stem 제거
        source_stem = stemmer.stem(word)
        source_lemma = lemmatiser.lemmatize(word, pos)
        for i, token in enumerate(list_synsets):
            #print(source_stem, stemmer.stem(str(token)), source_lemma, lemmatiser.lemmatize(str(token), pos))
            if source_stem == stemmer.stem(token):
                list_synsets[i] = 'to-be-deleted'
            if source_lemma == lemmatiser.lemmatize(token, pos):
                list_synsets[i] = 'to-be-deleted'
            if len(token.split('_')) != 1: # 'bill_of_exchange'와 같이 명사구 또는 복합명사일 경우... 삭제...
                list_synsets[i] = 'to-be-deleted'
        list_synsets = [x for x in list_synsets if x != 'to-be-deleted'] # 한번에 삭제      
        return list_synsets   
    
 
    def word2vec(self, word, pos):
        thr = sim_thr
        thr = thr / 2.0
        if type(word) != 'str':
            word = str(word)
        n_candidates = n_candidates_replace
        nlp_word = nlp.vocab[word]
        
        ### similarity가 비슷한 단어들 추출
        queries = [w for w in nlp_word.vocab if w.is_lower == nlp_word.is_lower and w.prob >= -11]
        by_similarity = sorted(queries, key=lambda w: nlp_word.similarity(w), reverse=True)
        #print(len([w.lower_ for w in by_similarity[:]]))
        #print(([w.lower_ for w in by_similarity[:]]))
        cand_word_list = [w.lower_ for w in by_similarity[:n_candidates]] # some candidate words
        #print(cand_word_list)
        
        ### 후보군들 중에서 어떤 것을 교체어로 선정할 것인가?
        # stemming과 lemmatization을 모두 사용하자.
        source_stem = stemmer.stem(word)
        source_lemma = lemmatiser.lemmatize(word, pos)
        
        for i, token in enumerate(cand_word_list):
            #print(source_stem, stemmer.stem(str(token)), source_lemma, lemmatiser.lemmatize(str(token), pos))
            if source_stem == stemmer.stem(token):
                cand_word_list[i] = 'to-be-deleted'
            if source_lemma == lemmatiser.lemmatize(token, pos):
                cand_word_list[i] = 'to-be-deleted'
        cand_word_list = [x for x in cand_word_list if x != 'to-be-deleted'] # 한번에 삭제

        #print(cand_word_list)
        #print('\n')
        if len(cand_word_list) == 0:
            return False
        if nlp_word.similarity(nlp.vocab[cand_word_list[0]]) > thr: # 70%로 비슷하면 
            return cand_word_list[0] # 최상위 1개 선택
            #return random.choice(cand_word_list) # 랜덤으로 선택
        else:
            list_wordnet = self.wordnet(word, pos)
            cand_word_list_filtered_from_wornet = [v for v in cand_word_list if v in list_wordnet]
            if len(cand_word_list_filtered_from_wornet)==0:
                if len(list_wordnet)==0:
                    return False
                else:
                    return list_wordnet[0]
            else:
                #print(cand_word_list_filtered_from_wornet)
                #print('\n')
                #return random.choice(cand_word_list_filtered_from_wornet)
                return cand_word_list_filtered_from_wornet[0]    

    def replace_algorithm(self, list_sent_raw, list_rep, pos):
        bit = False
        new_list_sent_raw = list_sent_raw[:]
        for token, _, idx in list_rep:
            result = self.word2vec(token, pos)
            if result != False: # 실패하지 않은 경우에만...
                bit = True
                new_list_sent_raw[idx] = result
        return new_list_sent_raw, bit

    def window_filtering(self, sent_label, window_size=3):
        list_sent_label = sent_label.split()
        new_list = [0] * len(list_sent_label)
        ### 순방향
        for i, _ in enumerate(list_sent_label):
            temp = list_sent_label[i:i+1+window_size]
            #print(temp)
            tkn = False
            for each in temp:
                if each != 'O':
                    tkn = True
            if tkn == True:
                new_list[i] = 1
        ### 역방향
        for i  in range(len(list_sent_label)-1, -1, -1):
            idx = i-window_size
            if idx < 0:
                idx = 0
            temp = list_sent_label[idx:i]
            tkn = False
            for each in temp:
                if each != 'O':
                    tkn = True
            if tkn == True:
                new_list[i] = 1      
        return new_list    
    
    def generation_via_replace(self, sent_raw, sent_label, alpha):

        #############
        """ 전처리 """
        #############
        # 동사, 명사는 context 정보의 핵심이다 모든 언어가 공통적으로 가지고 있는 특징.  
        #print(sent_raw)
        list_sent_raw = sent_raw.split()
        sent_raw = preprocessing_for_spacy(sent_raw)
        sent_nlp = nlp(sent_raw)
        noun_sorted, verb_sorted, adj_sorted, adv_sorted = [], [], [], []
        
        """
        ### assert 에러 났을 때, 원인 찾기 위해 print 실시
        if not len(sent_nlp) == len(sent_label.split()):
            print(len(sent_raw.split()), len(sent_nlp), len(sent_label.split()))
            print(sent_raw)
            print(sent_nlp)
            for i, token in enumerate(sent_nlp):
                print(token, sent_raw.split()[i], sent_label.split()[i])
        assert(len(sent_nlp) == len(sent_label.split()))
        """
        ### filtering with window size based on entities
        filtered_list = self.window_filtering(sent_label, window_size_replace)
        
        if len(sent_nlp) != len(sent_label.split()):
            #print('[ERROR]: len(list_sent_nlp) != len(list_sent_label)')
            return False    
        
        ### 조건에 맞는 token들을 불러온다 
        for i, token in enumerate(sent_nlp):
            ### 제외되는 token들..
            if not sent_label.split()[i] == 'O': # 엔티티가 아닌 token은 제외
                continue
            is_uppercase_letter = True in map(lambda l: l.isupper(), str(token))
            if i != 0 and is_uppercase_letter == True: # 첫 번째 글자가 아니고 대문자가 하나라도 있는 token은 제외 (ex. Thursday와 같은 시간고유명사. 크게 의미적이지 않음.)
                continue
            if lemmatiser.lemmatize(str(token), 'v') == 'be': # be동사 lemma를 가지는 token은 제외
                continue
            if lemmatiser.lemmatize(str(token), 'v') == 'have': # have 동사 lemma를 가지는 token은 제외, (have 동사의 의미가 광범위하게 쓰이기도 하고, have가 분사형으로 사용되기도 하기 때문이다)
                continue
            if filtered_list[i] == 0: # filtering with window size based on entities
                continue
                
            ### 동사, 명사, 형용사, 부사 추출
            if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                pair = (sent_nlp[i], len(list(token.subtree)), i) 
                noun_sorted.append(pair) # 명사
                   
            elif token.pos_ == 'VERB':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                verb_sorted.append(pair) # 동사       
            
            elif token.pos_ == 'ADJ':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                adj_sorted.append(pair) # 형용사         

            elif token.pos_ == 'ADV':
                pair = (sent_nlp[i], len(list(token.subtree)), i)
                adv_sorted.append(pair) # 부사   
        
        ### dependency 크기로 정렬
        noun_sorted.sort(key=lambda x: x[1], reverse=True)
        verb_sorted.sort(key=lambda x: x[1], reverse=True)
        adj_sorted.sort(key=lambda x: x[1], reverse=True)
        adv_sorted.sort(key=lambda x: x[1], reverse=True)
        
    #     print((noun_sorted))
    #     print((verb_sorted))
        if len(noun_sorted)==0 and len(verb_sorted)==0 and len(adj_sorted)==0 and len(adv_sorted)==0:
            return False # 교체 후보 token이 아예없다면, 새로운 문장을 만들 수 없다.

        ### thr 정하기
        noun_thr = int(len(noun_sorted) * alpha)
        verb_thr = int(len(verb_sorted) * alpha)
        adj_thr = int(len(adj_sorted) * alpha)
        adv_thr = int(len(adv_sorted) * alpha)
        
        if noun_thr == 0:
            noun_thr = 1 # 최소값은 1로 유지
        if verb_thr == 0:
            verb_thr = 1 # 최소값은 1로 유지
        if adj_thr == 0:
            adj_thr = 1 # 최소값은 1로 유지
        if adv_thr == 0:
            adv_thr = 1 # 최소값은 1로 유지        
        
        ### token list to be replaced
        noun_rep = noun_sorted[:noun_thr] 
        verb_rep = verb_sorted[:verb_thr] 
        adj_rep = adj_sorted[:adj_thr]
        adv_rep = adv_sorted[:adv_thr]
        
        #print(noun_rep)
        #print(verb_rep)
        #print(list_sent_raw)
        
        if len(noun_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_n = self.replace_algorithm(list_sent_raw, noun_rep, 'n')
            #print(list_sent_raw)
            #replaced_noun = (noun_sorted[0][0], noun_sorted[0][2]) # (token, index) pair 
            # 명사
            #print('<<< ', replaced_noun[0], '  >>>')
            #target_noun = replace_with_word2vec(replaced_noun[0], 'n')
            #print(replace_with_sense2vec(replaced_noun[0], 'NOUN'))
            #print(replace_with_wordnet(replaced_noun[0], 'n'))
            #print('replaced_noun: ', replaced_noun, ', target_noun: ', target_noun)
            #list_sent_raw[replaced_noun[1]] = target_noun # 교체
        else:
            bit_n = False
        if len(verb_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_v = self.replace_algorithm(list_sent_raw, verb_rep, 'v')
            #print(list_sent_raw)
            #replaced_verb = (verb_sorted[0][0], verb_sorted[0][2])
            # 동사
            #print('<<< ', replaced_verb[0], '  >>>')
            #target_verb = replace_with_word2vec(replaced_verb[0], 'v')
            #print(replace_with_sense2vec(replaced_verb[0], 'VERB'))
            #print(replace_with_wordnet(replaced_verb[0], 'v'))   
            #print('replaced_verb: ', replaced_verb, ', target_verb: ', target_verb)
            #list_sent_raw[replaced_verb[1]] = target_verb # 교체
        else:
            bit_v = False
        if len(adj_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_a = self.replace_algorithm(list_sent_raw, adj_rep, 'a')
        else:
            bit_a = False    
        if len(adv_sorted) != 0: # 각자 0이 될 수도 있으니...
            list_sent_raw, bit_r = self.replace_algorithm(list_sent_raw, adv_rep, 'r')
        else:
            bit_r = False 
        if bit_n==False and bit_v==False and bit_a==False and bit_a==False:
            return False
        else:    
            #print(' '.join(list_sent_raw))
            return ' '.join(list_sent_raw)    
    
    def do(self, data):
        sent_raw = data[0]
        sent_label = data[1]
        print('>>> [Replace Start!]: (number of input sent = {})'.format(len(sent_raw)))
    
        new_sent_raw_list = []
        new_sent_label_list = [] 
        for i, row in enumerate(sent_raw):
            alpha = 0.9
            result = self.generation_via_replace(sent_raw[i], sent_label[i], alpha)
            if result == False: # 교체어가 없었는 경우
                continue
            else:
                new_sent_raw_list.append(result)
                new_sent_label_list.append(sent_label[i]) # 라벨은 동일하다.    
        print('>>> [Replace Done!] (number of output (new) sent = {}'.format(len(new_sent_raw_list)))
        return new_sent_raw_list, new_sent_label_list, len(new_sent_raw_list)
        

        
        
