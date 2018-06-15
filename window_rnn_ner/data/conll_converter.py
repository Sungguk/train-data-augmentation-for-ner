## 4 column을 2 column으로
import os
"""
def load_dataset(fname):
    docs = []
    with open(fname, encoding="utf-8") as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split("\t",1))
        # flush running buffer
        docs.append(cur)
    return docs
""" 
    
def data_converter():

    path = os.getcwd()
    path += '\\conll2003' # 접근할 폴더 이름.
    
    for filename in os.listdir(path):
        full_path = path + '\\' + filename
        
        temp = []
        with open('converted_conll2003'+'\\'+filename, 'w') as wf:
            with open(full_path, 'r') as rf:
                for line in rf:
                    split_list = line.strip().split()
                    #print(split_list, len(split_list))
                    if len(split_list)==0:
                        wf.write('\n')
                    else:
                        temp_str = str(split_list[0]) + '\t' + str(split_list[-1])
                        wf.write(temp_str)
                        wf.write('\n')
                        
                    
        
        #file = open(full_path, 'r')
        #contents = file.readlines()
        #print(filename)
        #break
    
if __name__ == "__main__":
    data_converter()