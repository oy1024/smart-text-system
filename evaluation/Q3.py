import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg as pseg
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def clean_blank(title):
    """清理新闻标题空白"""
    # 英文大写转小写
    title = title.upper()
    # 清理未知字符
    title = re.sub(r'\?+', ' ', title)
    # 清理空白字符
    title = re.sub(r'\u3000', '', title)
    title = title.strip()
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'([|：])+ ', r'\1', title)
    title = re.sub(r' ([|：])+', r'\1', title)
    return title

def getF1(text):
    # 1. 匹配法律法规
    pattern1 = '((根据|按照|依照|依据).*((《.*》)|(政府令|政府发文)))'
    # 2. 匹配相关部门核实
    pattern2 = '(经.*(核实|调查|了解|核查|查实))'
    F1 = 0
    if re.search(pattern1,text):
        F1+=1
    if re.search(pattern2,text):
        F1+=1
    return F1

def getF2(text):
    return np.log10(len(text)) if np.log10(len(text))>=0 else 0

def getF3(x):
    return -np.log(np.log10(x)) if x>10 else 0

def test(str,d1,d2):
    str  = clean_blank(str)
    F1 = getF1(str)
    F2 = getF2(str)
    days = (d2-d1).days
    F3 = getF3(days) 

    return F1+F2+F3