import os
import re
import math
import jieba
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from itertools import product,count

def cut_sents(content):
    sentences = re.split(r"([。!！?？；;\s+])", content)[:-1]
#     sentences = re.split(r"")
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
    return sentences

def cut_word_test(context):
    stopkey=[line.strip() for line in open('customer/stopwords.txt',encoding='utf-8').readlines()] 
    total_cutword = []
    total_content = []
    for i in context:
        words=jieba.cut(i)
        words_filter=[word for word in words if word not in stopkey]
        if len(words_filter) !=0:
            total_cutword.append(words_filter)
            total_content.append(i)
    return total_cutword,total_content

def filter_model(sents,model):
    '''
    过滤词汇表中没有的单词
    '''
    total = []
    for sentence_i in sents:
        sentence_list = []
        for word_j in sentence_i:
            if word_j in model:
                sentence_list.append(word_j)
        total.append(sentence_list)
    return total


def two_sentences_similarity(sents_1,sents_2):
    '''
    计算两个句子的相似性
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter +=1
    return counter / (math.log(len(sents_1) + len(sents_2)))

def cosine_similarity(vec1,vec2):
    '''
    计算两个向量之间的余弦相似度
    '''
    tx =np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1/float(cos21 * cos22)
    return cosine_value

def computer_similarity_by_avg(sents_1,sents_2,model):
    '''
    对两个句子求平均词向量
    '''
    if len(sents_1) ==0 or len(sents_2) == 0:
        return 0.0
    vec1_avg = sum(model[word] for word in sents_1) / len(sents_1)
    vec2_avg = sum(model[word] for word in sents_2) / len(sents_2)
        
    similarity = cosine_similarity(vec1_avg , vec2_avg)
    return similarity

def create_graph(word_sent,model):
    '''
    传入句子链表，返回句子之间相似度的图
    '''
    num = len(word_sent)
    board = np.zeros((num,num))
    
    for i,j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = computer_similarity_by_avg(word_sent[i], word_sent[j],model)
    return board

def sorted_sentence(graph,sentences,topK):
    '''
    调用pagerank算法进行计算，并排序
    '''
    key_index = []
    key_sentences = []
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank_numpy(nx_graph)
#     sorted_scores = scores.items()
    sorted_scores = sorted(scores.items(), key = lambda item:item[1],reverse=True)
    for index,_ in sorted_scores[:topK]:
        key_index.append(index)
    new_index = sorted(key_index)
    for i in new_index:
        key_sentences.append(sentences[i])
    return key_sentences


def do(text,topK):
    list_sents = cut_sents(text)
    data,sentences = cut_word_test(list_sents) 
    # 训练模型
    model = Word2Vec(data, size=256, window=5,iter=10, min_count=1, workers=4)
    sents2 = filter_model(data,model)
    graph = create_graph(sents2,model)
    result_sentence = sorted_sentence(graph,sentences,topK)
    return "".join(result_sentence)

text = '原标题：专访：俄方希望与中方寻找双边贸易新增长点——访俄罗斯工业和贸易部长曼图罗夫\
新华社记者栾海高兰\
“在当前贸易保护主义抬头背景下，俄方希望与中方共同应对风险，化消极因素为机遇，寻找俄中贸易的新增长点”，俄罗斯工业和贸易部长丹尼斯·曼图罗夫日前在接受新华社记者专访时说。\
曼图罗夫表示，中国一直是俄重要的战略协作伙伴。当前俄中关系保持快速发展，双方不断在贸易和工业领域寻找新的合作点。据他介绍，今年1月至7月，俄中双边贸易额同比增长超25%，达近600亿美元。\
曼图罗夫说，俄中两国正在飞机轮船和其他交通工具制造、无线电设备研发、制药和化工等工业领域开展合作。俄中投资基金支持了两国众多开发项目，投资方对该基金继续注资的兴趣十分浓厚。\
在回顾日前结束的第四届东方经济论坛时，曼图罗夫表示，这一论坛已成为俄与中国和其他东北亚国家讨论重大经济合作议题的平台。“在本届论坛期间，俄方与海外企业共签署220项各类协议，协议总金额达3.1万亿卢布（1美元约合66卢布）”。\
曼图罗夫说，俄工业和贸易部在本届论坛上与俄外贝加尔边疆区的一家矿业公司负责人进行磋商，以落实中方企业持有该公司股份的相关事宜。根据相关协议，俄中企业将在外贝加尔边疆区的金矿区联合勘探。据俄方估算，这一俄中合作项目有望年产黄金约6.5吨，在2020年前使该边疆区贵金属开采量比目前增加约40%，从而有力促进当地经济发展。\
责任编辑：张义凌'

result_data = do(text,3)
print(result_data)
