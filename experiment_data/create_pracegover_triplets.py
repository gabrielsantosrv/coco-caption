import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('pracegover_duplications.json') as file:
    data = json.load(file)

selected_sents = []
count = 0
for k, v in data.items():
    sents = []
    for item in v:
        sents.append(item['caption'])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sents)
    matrix = cosine_similarity(X, X)
    sent_to_remove = []
    for i in range(1, matrix.shape[0]):
        if matrix[0, i] > 0.5:
            sent_to_remove.append(i)
    sents = [s for i, s in enumerate(sents) if i not in sent_to_remove]
    if len(sents) > 1:
        selected_sents.append(sents)

triplets = {'HCI':{}, 'HII':{}}
n = len(selected_sents)
index = 0
for i, sents in enumerate(selected_sents):
    n_sent = len(sents)
    for p in range(n_sent):
        for q in range(p+1, n_sent):
            j = i
            while j == i:
                j = random.randint(0, n-1)
            random_sent = random.choice(selected_sents[j])
            triplets['HCI'][index] = [sents[p], sents[q], random_sent, 1]

            words = sents[q].split()
            m = len(words)
            end = random.randint(m//4, m//2)
            triplets['HII'][index] = [sents[p], ' '.join(words[:end]), random_sent, 1]
            index += 1

print(len(triplets))
with open('pracegover_triplets.json', 'w') as file:
    json.dump(triplets, file)
