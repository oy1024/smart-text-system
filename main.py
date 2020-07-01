import streamlit as st
import datetime
import pandas as pd
from PIL import Image
import time
import sys
sys.path.append('fenlei/code')
sys.path.append('hot_mining')
sys.path.append('evaluation')
sys.path.append('Emotion_analysis')
from getLabel_new import getLabel
from hot import getHot
from Q3 import test
from Opinion_extraction import opinon_extra
from nlp import emotion_analysis

image = Image.open('1.png')
st.image(image, use_column_width =True, format='PNG' )
st.sidebar.header('智慧文本系统')



# image = Image.open('image.jpg')
# st.sidebar.image(image, 
#           use_column_width=True)

option = st.sidebar.selectbox('请选择要进行的操作',('NULL','文本评价', '文本分类', '热点挖掘','情感倾向分析','评论观点抽取'),index = 0)

if option == 'NULL':
    st.title('智慧文本挖掘系统')
    st.subheader('* 文本评价')
    st.markdown('对于一段回复文本，从答复的相关性、完整性、可解释性、时效性等多个角度量化文本，实现对文本的公正评价')
    st.subheader('* 文本分类')
    st.markdown('对于文本数据所属的类别进行种类识别并划分')
    st.subheader('* 热点挖掘')
    st.markdown('对文本数据进行聚类，并从大量的文本数据中发现热点问题，展示热点问题的词云图')
    st.subheader('* 情感倾向分析')
    st.markdown('从多维度对文本数据进行情感倾向的分析，实现文本感情极性的判断，并给出该情感的置信度')
    st.subheader('* 评论观点抽取')
    st.markdown('对文本、评论数据的关注点、评论观点自动分析，输出评论观点标签及观点极性')

if option == '文本评价':
    st.title('回复评价系统')
    st.markdown('Government response evaluation system')
    text = st.text_area('请输入需要评价的文本')
    d1 = st.date_input('留言时间',datetime.date(2019, 7, 6))
    d2 = st.date_input('回复时间',datetime.date(2019, 7, 6))
    if st.button('开始评价'):
        st.write('回复评分为', test(text,d1,d2))
        if test(text,d1,d2)>=4.8:
            st.write('优秀！回复内容充实，有法律依据')
        elif test(text,d1,d2)>=3.9:
            st.write('优秀！回复内容充实，回复较为及时')
        elif test(text,d1,d2)>=3.0:
            st.write('良好！处理高效，回复及时')
        elif test(text,d1,d2)>=2.1:
            st.write('良好！回复内容较为空洞，逻辑较差')
        elif test(text,d1,d2)>=1.2:
            st.write('不合格！回复内容少，没有逻辑')
        else:
            st.write('不合格！处理效率低，回复时间间隔长')

if option == '文本分类':
    st.title('文本分类系统')
    st.markdown('Message Text Classification System')
    text = st.text_area('请输入需要分类的文本')
    if st.button('开始分类'):
        st.write('类别为', getLabel(text))


if option == '热点挖掘':
    st.title('热点挖掘系统')
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data)
        if st.button('开始挖掘热点'):
            '聚类中请稍后...'
            df = getHot(data, 0.6, 10)
            # Add a placeholder
            st.write('聚类完成！')
            st.write(df)


if option == '情感倾向分析':
    st.title('情感分析系统')
    st.markdown('Affective tendency analysis system')
    text = st.text_area('请输入要分析的文本')
    if st.button('开始分析'):
        st.write('分析结果为') #展示一个感情置信度
        a,b,c,d = emotion_analysis(text)
        chart_data = pd.DataFrame([[c,d]],columns = ['positive_prob','negative_prob'])
        st.bar_chart(chart_data)
        if a==2:
            st.write('正向感情')
            st.write('分类置信度为：',b)
        elif a==1:
            st.write('中性感情')
            st.write('分类置信度为：',b)
        else:
            st.write('负向感情')
            st.write('分类置信度为：', b)






if option == '评论观点抽取':
    st.title('观点抽取系统')
    st.markdown('Viewpoint extraction system')
    op = st.selectbox('请选择文本的类别', ('酒店', 'KTV', '丽人', '美食餐饮', '旅游', '健康', '教育', '商业', '房产', '汽车', '生活', '购物', '3C'),index=0)
    text = st.text_area('请输入要抽取观点的文本')
    if st.button('观点抽取'):
        result_opinon = opinon_extra(text, op)
        st.write('评论观点为')
        st.write(result_opinon)
