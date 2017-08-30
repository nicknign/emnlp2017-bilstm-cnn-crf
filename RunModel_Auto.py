#!/usr/bin/python
#Usage: python RunModel.py modelPath inputPath"
from __future__ import print_function
import jieba as jb
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import sys
import logging

# if len(sys.argv) < 3:
#    print("Usage: python RunModel.py modelPath inputPath")
#    exit()
# modelPath = sys.argv[1]
# inputPath = sys.argv[2]

# :: Logging level ::
loggingLevel = logging.DEBUG
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

# :: init ::
jb.load_userdict("util/fenci.txt")

modelPath = "models/AutoForce/NER_BIO/1.0000_1.0000_23.h5"
inputPath = "input.txt"

with open(inputPath, 'r') as f:
    textlines = f.readlines()


# :: Load the model ::
lstmModel = BiLSTM()
lstmModel.loadModel(modelPath)


# :: Prepare the input ::
sentences = [{'tokens': [sent for sent in jb.cut(line.strip('\n'))]} for line in textlines]
addCharInformation(sentences)
addCasingInformation(sentences)

dataMatrix = createMatrices(sentences, lstmModel.mappings)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']
    tokenTags = tags[sentenceIdx]
    for tokenIdx in range(len(tokens)):
        print("%s\t%s" % (tokens[tokenIdx], tokenTags[tokenIdx]))
    print("")

while(1):
    sentence = raw_input(">>>>>>>>")
    sentences = [{'tokens': [sent for sent in jb.cut(sentence)]}]
    addCharInformation(sentences)
    addCasingInformation(sentences)
    dataMatrix = createMatrices(sentences, lstmModel.mappings)
    tags = lstmModel.tagSentences(dataMatrix)
    tokens = sentences[0]['tokens']
    tokenTags = tags[0]
    for tokenIdx in range(len(tokens)):
        print("%s\t%s" % (tokens[tokenIdx], tokenTags[tokenIdx]))
    print("")
