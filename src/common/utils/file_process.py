
import os, sys
import json

#递归遍历文件夹下所有的文件名
def find_all_files(dir_path, file_name_list):
    if os.path.isfile(dir_path):
        file_name_list.append(dir_path)
        return
    else:
        paths_here = os.listdir(dir_path)
        for a_path in paths_here:
            a_path = dir_path + '/' + a_path
            if os.path.isfile(a_path):
                file_name_list.append(a_path)
            else:
                find_all_files(a_path, file_name_list)
                
def load_corpus(file_name):
    lines = list(open(file_name, 'r', encoding='utf8').readlines())
    lines = list(map(lambda x: json.loads(x.strip()), lines))
    return lines
                
if __name__ == '__main__':
    files = []
    find_all_files('../../data/stop_words', files)
    print(files)
    
    