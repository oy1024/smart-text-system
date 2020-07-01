from aip import AipNlp
import json

""" 你的 APPID AK SK """
APP_ID = '20634799'
API_KEY = 'pvqKBQ7Y8VxiIPN7jZLGqwwO'
SECRET_KEY = 'STfqCm8qaGaHWCm1nEeWPGRr2ahh05Ku'


def emotion_analysis(text):
    client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
    #text = "苹果是一家伟大的公司"
    """ 调用情感倾向分析 """
    ans = client.sentimentClassify(text)

    return ans['items'][0]['sentiment'], ans['items'][0]['confidence'], ans['items'][0]['positive_prob'], ans['items'][0]['negative_prob']
