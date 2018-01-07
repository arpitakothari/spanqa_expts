import json
import operator
import numpy as np
import matplotlib.pyplot as plt

origin = 'squad'
num_top = 20

f = open(origin+'_dev.json', 'r')

biword = {}
total = 0

for row in f:
    row = json.loads(row)
    quest = row['question']
    quest_words = quest.split()
    curr = quest_words[0].lower() + " " + quest_words[1].lower()
    if curr in biword:
        biword[curr] += 1
    else:
        biword[curr] = 1
    total += 1

sorted_biword = sorted(biword.items(), key=operator.itemgetter(1), reverse=True)
biword_stats = sorted_biword[:num_top]

fig, ax = plt.subplots()
ind = np.arange(num_top)

stats = [(x[1]*100)/float(total) for x in biword_stats]
labels = [str(x[0]) for x in biword_stats]

# print stats
# print labels

plt.bar(ind, stats)
plt.xticks(ind, labels, rotation='vertical')
ax.set_xlabel('Starting bi-word')
ax.set_ylabel('Percent questions')

plt.tight_layout()
plt.savefig(origin+'_dev_quest_distr.jpg')

# print total
