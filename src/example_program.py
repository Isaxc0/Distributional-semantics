import pandas as pd
import matplotlib.pyplot as plt
from itertools import zip_longest
import word_vector
import distributional_semantics

"""
EXAMPLE PROGRAM

Displays the 10 most similar words to the most frequent word according to similairty score
"""
sentences = normalise(tokenise())
all_words = []
for sentence in sentences:
  for word in sentence:
    all_words.append(word)
all_words = set(all_words)
all_word_vectors = word_vector(all_words,1)
top_word = find_frequently_occurring_words(sentences)[0] #the most frequent word


#finds the cosine similarity between the top frequeny word and every other word
dist_top_word_sims = {}
#iterates over each word in every sentence allowing each combination to be tried
for word in all_words:
  if word != top_word:
    similarity = all_word_vectors.similarity(top_word, word)
    dist_top_word_sims[word] = similarity

dist_top_word_sims_sorted = dict(sorted(dist_top_word_sims.items(), key=lambda item: item[1],reverse=True)) #sorts dictionary of word frequency from largest to smallest similarity score
dist_similar_words = list(dist_top_word_sims_sorted.keys())[:10] #top 10 most similar words to the most frequent word
dist_similar_values = list(dist_top_word_sims_sorted.values())[:10] #top 10 most similar values to the most frequent word

#displaying results
print("Most frequent word: '{}'".format(top_word))
df = pd.DataFrame(zip(dist_similar_words,dist_similar_values),columns=["Top 10 most similar words","Score"],index=["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"])
display(df)