import pandas as pd
import numpy as np
import re
import time
import jieba
import jieba.posseg as pseg
import json
from datetime import datetime
from datetime import timedelta
from collections import Counter

def data_filter(df):
    """数据过滤"""
    # 过滤掉没有内容的新闻
    df = df[df['content'] != ''].copy()
    df = df.dropna(subset=['content']).copy()
    # 去重
    #df = df.drop_duplicates(subset=['title'])
    df = df.reset_index(drop=True)
    return df


def get_data(df, last_time, delta):
    """
    获取某段时间的新闻数据
    :param df: 原始数据
    :param last_time: 指定要获取数据的最后时间
    :param delta: 时间间隔
    :return: last_time前timedelta的数据
    """
    last_time = datetime.strptime(last_time, '%Y/%m/%d %H:%M:%S')
    delta = timedelta(delta)
    try:
        df['time'] = df['time'].map(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S'))
    except TypeError:
        pass
    df = df[df['time'].map(lambda x: (x <= last_time) and (x > last_time - delta))].copy()
    print('df.shape=', df.shape)
    if df.shape[0] == 0:
        print('No Data!')
        return df
    df = df.sort_values(by=['time'], ascending=[0])
    df['time'] = df['time'].map(lambda x: datetime.strftime(x, '%Y/%m/%d %H:%M:%S'))
    df = df.reset_index(drop=True)
    return df


def clean_title_blank(title):
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


def clean_content_blank(content):
    """清理新闻内容空白"""
    # 清理未知字符
    content = re.sub(r'\?+', ' ', content)
    # 清理空白字符
    content = re.sub(r'\u3000', '', content)
    content = content.strip()
    content = re.sub(r'[ \t\r\f]+', ' ', content)
    content = re.sub(r'\n ', '\n', content)
    content = re.sub(r' \n', '\n', content)
    content = re.sub(r'\n+', '\n', content)
    return content


def clean_content(content):
    """清理新闻内容"""
    # 清理新闻内容空白
    content = clean_content_blank(content)
    # 英文大写转小写
    content = content.upper()
    # 清理超链接
    content = re.sub(r'https?://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', '', content)
    # 清理来源等和内容无关的文字
    texts = []
    for text in texts:
        content = re.sub(text, '', content)
    content = re.sub(r'\n+', '\n', content)
    return content


def get_num_en_ch(text):
    """提取数字英文中文"""
    text = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF]+', ' ', text)
    text = text.strip()
    return text


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


def userdict_cut(text, userdict_path=None):
    """
    对文本进行jieba分词
    如果使用用户词词典，那么使用用户词词典进行jieba分词
    """
    if userdict_path is not None:
        jieba.load_userdict(userdict_path)
    words = jieba.cut(text)
    return words


def stop_words_cut(words, stop_words_path):
    """停用词处理"""
    with open(stop_words_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
        stopwords.append(' ')
        words = [word for word in words if word not in stopwords]
    return words


def disambiguation_cut(words, disambiguation_dict_path):
    """消歧词典"""
    with open(disambiguation_dict_path, 'r', encoding='utf-8') as f:
        disambiguation_dict = json.load(f)
        words = [(disambiguation_dict[word]
                  if disambiguation_dict.get(word) else word) for word in words]
    return words


def individual_character_cut(words, individual_character_dict_path):
    """删除无用单字"""
    with open(individual_character_dict_path, 'r', encoding='utf-8') as f:
        individual_character = [line.strip() for line in f.readlines()]
        words = [word for word in words
                 if ((len(word) > 1) or ((len(word) == 1) and (word in individual_character)))]
    return words


def document2txt(raw_document, userdict_path, text_path):
    """文本分词并保存为txt文件"""
    document = clean_content_blank(raw_document)
    document = document.upper()
    document_cut = userdict_cut(document, userdict_path)
    result = ' '.join(document_cut)
    result = re.sub(r' +', ' ', result)
    result = re.sub(r' \n ', '\n', result)
    with open(text_path, 'w+', encoding='utf-8') as f:
        f.write(result)



def flat(l):
    """平展多维列表"""
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)


def get_word_library(list1):
    """
    获得词库
    :param list1: 一维或多维词列表
    :return: list，所有词去重之后的列表
    """
    list2 = flat(list1)
    list3 = list(set(list2))
    return list3


def get_single_frequency_words(list1):
    """
    获得单频词列表
    :param list1: 一维或多维词列表
    :return: list，所有只出现一次的词组成的列表
    """
    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i for i in cnt if cnt[i] == 1]
    return list3


def get_most_common_words(list1, top_n=None, min_frequency=1):
    """
    获取最常见的词组成的列表
    :param list1: 一维或多维词列表
    :param top_n: 指定最常见的前n个词，默认为None
    :param min_frequency: 指定最小频数，默认为1
    :return: list，最常见的前n个词组成的列表
    """
    list2 = flat(list1)
    cnt = Counter(list2)
    list3 = [i[0] for i in cnt.most_common(top_n) if cnt[i[0]] >= min_frequency]
    return list3


def get_num_of_value_no_repeat(list1):
    """
    获取列表中不重复的值的个数
    :param list1: 列表
    :return: int，列表中不重复的值的个数
    """
    num = len(set(list1))
    return num

import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from textrank4zh import TextRank4Sentence
from gensim.models import word2vec


def feature_extraction(series, vectorizer='CountVectorizer', vec_args=None):
    """
    对原文本进行特征提取
    :param series: pd.Series，原文本
    :param vectorizer: string，矢量化器，如'CountVectorizer'或者'TfidfVectorizer'
    :param vec_args: dict，矢量化器参数
    :return: 稀疏矩阵
    """
    vec_args = {'max_df': 1.0, 'min_df': 1} if vec_args is None else vec_args
    vec_args_list = ['%s=%s' % (i[0],
                                "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                ) for i in vec_args.items()]
    vec_args_str = ','.join(vec_args_list)
    vectorizer1 = eval("%s(%s)" % (vectorizer, vec_args_str))
    matrix = vectorizer1.fit_transform(series)
    return matrix


def get_cluster(matrix, cluster='DBSCAN', cluster_args=None):
    """
    对数据进行聚类，获取训练好的聚类器
    :param matrix: 稀疏矩阵
    :param cluster: string，聚类器
    :param cluster_args: dict，聚类器参数
    :return: 训练好的聚类器
    """
    cluster_args = {'eps': 0.5, 'min_samples': 5, 'metric': 'cosine'} if cluster_args is None else cluster_args
    cluster_args_list = ['%s=%s' % (i[0],
                                    "'%s'" % i[1] if isinstance(i[1], str) else i[1]
                                    ) for i in cluster_args.items()]
    cluster_args_str = ','.join(cluster_args_list)
    cluster1 = eval("%s(%s)" % (cluster, cluster_args_str))
    cluster1 = cluster1.fit(matrix)
    return cluster1


def get_labels(cluster):
    """
    获取聚类标签
    :param cluster: 训练好的聚类器
    :return: list，聚类标签
    """
    labels = cluster.labels_
    return labels


def label2rank(labels_list):
    """
    按标签的数量将标签转换为排行
    :param labels_list: list，聚类标签
    :return: list，聚类排行
    """
    series = pd.Series(labels_list)
    list1 = series[series != -1].tolist()
    n = len(set(list1))
    cnt = Counter(list1)
    key = [cnt.most_common()[i][0] for i in range(n)]
    value = [i for i in range(1, n + 1)]
    my_dict = dict(zip(key, value))
    my_dict[-1] = -1
    rank_list = [my_dict[i] for i in labels_list]
    return rank_list


def get_non_outliers_data(df, label_column='label'):
    """获取属于某个聚类簇的数据"""
    df = df[df[label_column] != -1].copy()
    return df


def get_data_sort_labelnum(df, label_column='label', top=1):
    """
    获取按标签数量排行的第top组数据
    :param df: pd.DataFrame，带有标签列的数据
    :param label_column: string，标签列名
    :param top: int
    :return: pd.DataFrame，前top组的数据
    """
    assert top > 0, 'top不能小于等于0！'
    labels = df[label_column].tolist()
    cnt = Counter(labels)
    label = cnt.most_common()[top - 1][0] if top <= len(set(labels)) else -2
    df = df[df[label_column] == label].copy() if label != -2 else pd.DataFrame(columns=df.columns)
    return df


def list2wordcloud(list1, save_path, font_path):
    """
    将文本做成词云
    :param list1: list，文本列表
    :param save_path: string，词云图片保存的路径
    :param font_path: string，用于制作词云所需的字体路径
    """
    text = ' '.join(list1)
    wc = WordCloud(font_path=font_path, width=800, height=600, margin=2,
                   ranks_only=True, max_words=200, collocations=False).generate(text)
    wc.to_file(save_path)


def get_key_sentences(text, num=1):
    """
    利用textrank算法，获取文本摘要
    :param text: string，原文本
    :param num: int，指定摘要条数
    :return: string，文本摘要
    """
    tr4s = TextRank4Sentence(delimiters='\n')
    tr4s.analyze(text=text, source='all_filters')
    abstract = '\n'.join([item.sentence for item in tr4s.get_key_sentences(num=num)])
    return abstract


def feature_reduction(matrix, pca_n_components=50, tsne_n_components=2):
    """降维"""
    data_pca = PCA(n_components=pca_n_components).fit_transform(matrix) if pca_n_components is not None else matrix
    data_pca_tsne = TSNE(n_components=tsne_n_components).fit_transform(
        data_pca) if tsne_n_components is not None else data_pca
    print('data_pca_tsne.shape=', data_pca_tsne.shape)
    return data_pca_tsne





def get_wordvec(model, word):
    """查询词是否在词库中"""
    try:
        model.wv.get_vector(word)
        return True
    except:
        return False


def get_word_and_wordvec(model, words):
    """获取输入词的词和对应的词向量"""
    word_list = [i for i in words if get_wordvec(model, i)]
    wordvec_list = [model.wv[i].tolist() for i in words if get_wordvec(model, i)]
    return word_list, wordvec_list


def get_top_words(words, label, label_num):
    """获得每个类中的前30个词"""
    df = pd.DataFrame()
    df['word'] = words
    df['label'] = label
    for i in range(label_num):
        df_ = df[df['label'] == i]
        print(df_['word'][:30])


def save_model(model, model_path):
    """保存模型"""
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(model_path):
    """加载模型"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


eps_var = 0.6
min_samples_var = 10
def getHot(data,eps_var,min_samples_var):
    df = data[[]]
    df['title'] = data['留言主题']
    df['content'] = data['留言详情']
    df['up'] = data['反对数']
    df['down'] = data['点赞数']


    # 数据清洗

    # 去除空白
    df['title_'] = df['title'].map(lambda x: clean_title_blank(x))
    df['content'] = df['content'].map(lambda x: clean_content(x))
    df['content_'] = df['content'].map(lambda x: get_num_en_ch(x))
    df['content_cut'] = df['content_'].map(lambda x: pseg_cut(
        x, userdict_path='hot_mining/data/extra_dict/self_userdict.txt'))
    df['content_cut'] = df['content_cut'].map(lambda x: get_words_by_flags(
        x, flags=['n.*', 'v.*', 'eng', 't', 's', 'j', 'l', 'i']))
    df['content_cut'] = df['content_cut'].map(lambda x: stop_words_cut(
        x, 'hot_mining/data/extra_dict/self_stop_words.txt'))
    df['content_'] = df['content_cut'].map(lambda x: ' '.join(x))
    word_library_list = get_word_library(df['content_cut'])
    single_frequency_words_list = get_single_frequency_words(df['content_cut'])
    # 存疑
    max_features = (len(word_library_list) - len(single_frequency_words_list)) // 2
    # 转换为TDIDF
    matrix = feature_extraction(df['content_'], vectorizer='TfidfVectorizer',
                                             vec_args={'max_df': 0.95, 'min_df': 1, 'max_features': max_features})

    # 建立模型 最大边距/ 最小个数
    dbscan = get_cluster(matrix, cluster='DBSCAN',
                        cluster_args={'eps': eps_var, 'min_samples': min_samples_var, 'metric': 'cosine'})
    labels = get_labels(dbscan)
    df['label'] = labels
    ranks = label2rank(labels)
    df['rank'] = ranks
    df_non_outliers = df[df['label'] != -1].copy()
    # 类别数量
    rank_num = get_num_of_value_no_repeat(df_non_outliers['rank'])
    print("共有",rank_num,"类")
    return df