import difflib
import pandas as pd

def match_things(key_word, name, num, rate):
    li_dis = load_enti_list('disease')
    li_micro = load_enti_list('micro')
    simi = difflib.get_close_matches(key_word, li_dis, num, cutoff=rate)
    if name == 'dis':
        simi = difflib.get_close_matches(key_word, li_dis, num, cutoff=rate)
    if name == 'micro':
        simi = difflib.get_close_matches(key_word, li_micro, num, cutoff=rate)
    simi.append('')
    return pd.DataFrame({name:simi})

def load_enti_list(name):#读取实体文件
        f = open(r'../qianduan/data/'+name+'.txt','r', encoding='utf-8')
        return f.read().splitlines()