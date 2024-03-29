from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
import jieba
import numpy as np
import jieba.posseg as pseg
import re
import xgboost as xgb
import lightgbm as lgb
import pandas as pd

f=open('fenlei/model/tf_idf_transformer','rb')
tf_idf_transformer=pickle.load(f)
f.close()
f=open('fenlei/model/vectorizer','rb')
vectorizer=pickle.load(f)
f.close()
NB_model = joblib.load('fenlei/model/NB_model')
xgb_model = xgb.Booster(model_file='fenlei/model/xgb_model')  # 加载训练好的xgboost模型
lgb_model = lgb.Booster(model_file='fenlei/model/lightGBM_model')  # 加载训练好的lightGBM模型

stopword = pd.read_csv('fenlei/data/extra_dict/self_stop_words.txt',encoding='utf-8',sep='never_exist',header = None)
T = {0:'城乡建设',1:'环境保护',2:'交通运输',3:'教育文体',4:'劳动和社会保障',5:'商贸旅游',6:'卫生计生'}


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

def pseg_cut(text, userdict_path=None):
    """
    词性标注
    :param text: string，原文本数据
    :param userdict_path: string，用户词词典路径，默认为None
    :return: list， 分词后词性标注的列表
    """
    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = pseg.lcut(text)
    return words

def get_num_en_ch(text):
    """提取数字英文中文"""
    text = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF]+', ' ', text)
    text = text.strip()
    return text

def stop_words_cut(words, stop_words_path):
    """停用词处理"""
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.append(' ')
        words = [word for word in words if word not in stopwords]
    return words

def get_words_by_flags(words, flags=None):
    """
    获取指定词性的词
    :param words: list， 分词后词性标注的列表
    :param flags: list， 词性标注，默认为提取名词和动词
    :return: list， 指定词性的词
    """
    flags = ['n.*', 'v.*'] if flags is None else flags
    words = [w for w, f in words if w != ' ' and re.match('|'.join(['(%s$)' % flag for flag in flags]), f)]
    return words



def getLabel(text):
    text = clean_blank(text)
    text = get_num_en_ch(text)
    text = jieba.lcut(text)
    text = [i for i in text if i not in stopword and len(i)>1]
    text = ' '.join(text)
    test = tf_idf_transformer.transform(vectorizer.transform([text])).toarray()
    dtest = xgb.DMatrix(test)
    xgb_pred = xgb_model.predict(dtest)
    lgb_pred = lgb_model.predict(test, num_iteration=lgb_model.best_iteration)
    NB_pred = NB_model.predict_proba(test)
    pred = (xgb_pred + lgb_pred + NB_pred) / 3
    predict = np.argmax(pred, axis=1)
    label = NB_model.predict(test)
    return T[label[0]]

