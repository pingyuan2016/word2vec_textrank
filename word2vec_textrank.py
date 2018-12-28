def cut_sents(content):
    sentences = re.split(r"([。!！?？；;\s+])", content)[:-1]
    sentences.append("")
    sentences = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]
    return sentences

def cut_word(context):
    stopkey=[line.strip() for line in open('customer/stopwords.txt',encoding='utf-8').readlines()] 
    total_cutword = []
    total_content = []
    for i in context:
        words=jieba.cut(i)
        words_filter=[word for word in words if word[0] not in stopkey]
        if len(words_filter) !=0:
            total_cutword.append(words_filter)
            total_content.append(i)
    return total_cutword,total_content

def filter_model(sents):
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

def computer_similarity_by_avg(sents_1,sents_2):
    '''
    对两个句子求平均词向量
    '''
    if len(sents_1) ==0 or len(sents_2) == 0:
        return 0.0
    vec1_avg = sum(model[word] for word in sents_1) / len(sents_1)
    vec2_avg = sum(model[word] for word in sents_2) / len(sents_2)
        
    similarity = cosine_similarity(vec1_avg , vec2_avg)
    return similarity

def create_graph(word_sent):
    '''
    传入句子链表，返回句子之间相似度的图
    '''
    num = len(word_sent)
    board = np.zeros((num,num))
    
    for i,j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = computer_similarity_by_avg(word_sent[i], word_sent[j])
    return board

def sorted_sentence(graph,sentences,topK):
    '''
    调用pagerank算法进行计算，并排序
    '''
    key_index = []
    key_sentences = []
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank_numpy(nx_graph)
    sorted_scores = sorted(scores.items(), key = lambda item:item[1],reverse=True)
    for index,_ in sorted_scores[:topK]:
        key_index.append(index)
    new_index = sorted(key_index)
    for i in new_index:
        key_sentences.append(sentences[i])
    return key_sentences


def do(text,topK):
    list_sents = cut_sents(text)
    data,sentences = cut_word(list_sents) 
    # 训练模型
    model = Word2Vec(data, size=256, window=5,iter=10, min_count=1, workers=4)
    sents2 = filter_model(data)
    graph = create_graph(sents2)
    result_sentence = sorted_sentence(graph,sentences,topK)
    print("关键句：")
    return "".join(result_sentence)
