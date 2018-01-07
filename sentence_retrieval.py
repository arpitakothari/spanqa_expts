import math
import nltk
import json
import operator
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()
stop = set(stopwords.words('english'))

def jaccard_similarity(question, passage):
    ret_dict = {}
    sent_index = 0
    for sent_words in passage:
        sent_words = [stemmer.stem(word.lower()) for word in sent_words]
        overlap = 0
        union = set(sent_words)
        for word in question:
            if word in sent_words:
                overlap += 1
            else:
                union.add(word)
        ret_dict[sent_index] = float(overlap)/len(union)
        sent_index += 1
    return ret_dict

def tf_idf(question, passage):
    ret_dict = {}
    sent_index = 0
    idf = {}
    tf = {}
    for sent_words in passage:
        sent_words = [stemmer.stem(word.lower()) for word in sent_words]
        for word in set(question):
            if sent_index not in tf:
                tf[sent_index] = {}
            tf[sent_index][word] = sent_words.count(word)
            if word not in idf:
                idf[word] = 0
            if tf[sent_index][word] > 0:
                idf[word] += 1
        sent_index += 1
    num_sentences = sent_index
    for i in range(num_sentences):
        ret_dict[i] = 0
        for word in question:
            ret_dict[i] += math.log(tf[i][word] + 1) * math.log(float(num_sentences)/(idf[word] + 1))
    return ret_dict

def bm_25(question, passage):
    k1 = 1.6
    b = 0.75
    ret_dict = {}
    sent_index = 0
    idf = {}
    tf = {}
    avgdl = 0
    for sent_words in passage:
        sent_words = [stemmer.stem(word.lower()) for word in sent_words]
        avgdl += len(sent_words)
        for word in set(question):
            if sent_index not in tf:
                tf[sent_index] = {}
            tf[sent_index][word] = sent_words.count(word)
            if word not in idf:
                idf[word] = 0
            if tf[sent_index][word] > 0:
                idf[word] += 1
        sent_index += 1
    num_sentences = sent_index
    avgdl = float(avgdl)/num_sentences
    for i in range(num_sentences):
        ret_dict[i] = 0
        for word in question:
            idf_curr = math.log((num_sentences - idf[word] + 0.5)/float(idf[word] + 0.5))
            score = idf_curr * (tf[i][word] * (k1+1))/float(tf[i][word] + k1*(1 - b + (b*len(passage[i])/avgdl)))
            ret_dict[i] += score
    return ret_dict

def eval_top(ret_dict, passag, answers, num_top):
    sorted_dict = sorted(ret_dict.items(), key=operator.itemgetter(1), reverse=True)
    ret_stats = sorted_dict[:num_top]
    result = []
    for item in ret_stats:
        sent = passag[item[0]]
        f = 0
        for answ in answers:
            if answ in sent:
                f = 1
        result.append(f)
    return result

def main():
    origin = 'newsqa'
    num_top = 2

    jac_sim_result = [0] * num_top
    tf_idf_result = [0] * num_top
    bm_25_result = [0] * num_top 

    f = open(origin+'_dev.json', 'r')
    total = 0

    for row in f:
        total += 1
        row = json.loads(row)
        quest = row['question']
        question = word_tokenize(quest)
        question = [stemmer.stem(word.lower()) for word in question]
        passg = row['passage']
        passag = sent_tokenize(passg)
        passage = [word_tokenize(sent) for sent in passag]
        answers = []
        answers = [answer['text'].strip() for answer in row['true_answers']]

        jac_sim_dict = jaccard_similarity(question, passage)
        res_1 = eval_top(jac_sim_dict, passag, answers, num_top)
        jac_sim_result = [x + y for x, y in zip(jac_sim_result, res_1)]
        
        tf_idf_dict = tf_idf(question, passage)
        res_2 = eval_top(tf_idf_dict, passag, answers, num_top)
        tf_idf_result = [x + y for x, y in zip(tf_idf_result, res_2)]

        bm_25_dict = bm_25(question, passage)
        res_3 = eval_top(bm_25_dict, passag, answers, num_top)
        bm_25_result = [x + y for x, y in zip(bm_25_result, res_3)]
        
    jac_sim_result = [elem/float(total) for elem in jac_sim_result]
    print jac_sim_result

    tf_idf_result = [elem/float(total) for elem in tf_idf_result]
    print tf_idf_result

    bm_25_result = [elem/float(total) for elem in bm_25_result]
    print bm_25_result

if __name__== '__main__':
    main()

