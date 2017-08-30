# -*- coding: utf-8 -*-
# !/usr/bin/env python
import sys
import os
import time
import json
import re
import tensorflow as tf
from aip import AipNlp
import pandas as pd

reload(sys)
sys.setdefaultencoding('utf8')

filepath = os.path.split(os.path.realpath(__file__))[0]

APP_ID = '9585999'
API_KEY = 'EHjkhXXsbLXATesvGBLu7PgB'
SECRET_KEY = 'tiTrCDQTqMQBbsarcn52CeP0Dc7bgmzT'


class Word(object):
    def __init__(self):
        self.reChinese = re.compile(u"[\u4e00-\u9fa5]+")
        self.aip = AipNlp(APP_ID, API_KEY, SECRET_KEY)
        self.vecdf = pd.read_csv("{}/vec.csv".format(filepath), encoding="utf_8")

    def word_sim(self, word1, word2):
        result = self.aip.wordembedding(unicode(word1), unicode(word2))
        sim = 0
        tf.logging.debug("wordsim result:{}".format(unicode(result)))
        if result.get('sim'):
            sim = result['sim']['sim']
        return sim

    def add_vec(self, word, vecjs):
        columns = ["word", "vec"]
        df = pd.DataFrame([[word, vecjs]], columns=columns)
        self.vecdf = self.vecdf.append(df, ignore_index=True)
        self.vecdf.to_csv("{}/vec.csv".format(filepath), encoding="utf_8", index=False)

    def word_vec(self, word):
        # tf.logging.debug("wordvec word:{}".format(unicode(word)))
        df = self.vecdf.loc[self.vecdf["word"] == word]
        if df.shape[0]:
            vec = df.iloc[0]["vec"]
            while isinstance(vec, unicode):
                vec = json.loads(vec)
            return vec

        result = {}
        for i in range(3):
            result = self.aip.wordembedding(unicode(word.lower()))
            tf.logging.warning("wordvec result:{}, word:{}".format(result, unicode(word)))
            if result.get('error_code') and result[u"error_code"] != 282130:
                time.sleep(1)
            else:
                break
        if result.get('vec'):
            vec = result['vec']['vec']
            vecjs = json.dumps(str(vec))
            self.add_vec(word, vecjs)
        else:
            vec = [0] * 128
        return vec

if __name__ == "__main__":
    wordc = Word()
    while(1):
        print(wordc.word_vec(raw_input(">>>")))
