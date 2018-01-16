import spacy
import json

nlp = spacy.load('en')

def check_np_ent(passg, answers):
    doc = nlp(passg)
    np_flag = 0
    ent_flag = 0
    for ans in answers:
        for np in doc.noun_chunks:
            if np.text in ans or ans in np.text:
                np_flag = 1
        for ent in doc.ents:
            if ans == ent.text:
                ent_flag = 1
    return np_flag, ent_flag

def main():
    origin = 'squad'

    np_total = 0
    ent_total = 0

    f = open(origin+'_dev.json', 'r')
    total = 0

    for row in f:
        total += 1
        row = json.loads(row)
        quest = row['question']
        passg = row['passage']
        answers = [answer['text'].strip() for answer in row['true_answers']]
        np_curr, ent_curr = check_np_ent(passg, answers)
        np_total += np_curr
        ent_total += ent_curr

    print np_total/float(total)
    print ent_total/float(total)

if __name__== '__main__':
    main()

