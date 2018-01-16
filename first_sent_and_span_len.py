import re
import nltk
import json
import operator
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

def check_first(answers, passage):
    for ans in answers:
        if ans[-1] in string.punctuation:
            ans = ans[:-1]
        if ans.lower() in passage[0].lower():
            return 1
    return 0

def span_len_avg(answers):
    num = 0
    len_total = 0
    for ans in answers:
        num += 1
        len_total += len(word_tokenize(ans))
    if num == 0:
        return 0
    return len_total/num

def main():
    origin = 'newsqa'

    first = 0
    span_dict = {}
    span_total = 0

    f = open(origin+'_dev.json', 'r')
    total = 0
    total_not_none = 0

    for row in f:
        total += 1
        row = json.loads(row)
        quest = row['question']
        passg = row['passage']
        passag = sent_tokenize(passg)
        answers = [answer['text'].strip() for answer in row['true_answers']]
        answers = set(answers)
        if 'none' in answers:
            answers.remove('none')
        if len(answers) != 0:
            total_not_none += 1
        first += check_first(answers, passag)
        span_curr = span_len_avg(answers)
        try:
            span_dict[span_curr] += 1
        except:
            span_dict[span_curr] = 1
        span_total += span_curr

    print first/float(total_not_none)
    print span_total/float(total_not_none)
    for span in span_dict:
        span_dict[span] /= float(total)

    sorted_span = sorted(span_dict.items(), key=operator.itemgetter(0))

    fig, ax = plt.subplots()
    ind = np.arange(len(sorted_span))

    stats = [x[1]*100 for x in sorted_span]
    labels = [str(x[0]) for x in sorted_span]

    plt.bar(ind, stats)
    plt.xticks(ind, labels, rotation='vertical')
    ax.set_xlabel('Answer span length')
    ax.set_ylabel('Percent questions')

    plt.tight_layout()
    plt.savefig(origin+'_dev_ans_len_distr.jpg')

if __name__== '__main__':
    main()
