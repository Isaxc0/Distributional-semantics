import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('wordnet_ic')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic as wn_ic

from sussex_nltk.corpus_readers import ReutersCorpusReader


#normalises tokens
def normalise(tokenlist):
    tokenlist=[token.lower() for token in tokenlist]
    tokenlist=["NUM" if token.isdigit() else token for token in tokenlist]
    tokenlist=["Nth" if (token.endswith(("nd","st","th")) and token[:-2].isdigit()) else token for token in tokenlist]
    tokenlist=["NUM" if re.search("^[+-]?[0-9]+\.[0-9]",token) else token for token in tokenlist]
    return tokenlist


#tokenises sentences
def tokenise():
    samplesize=2000
    iterations =100
    sentences=[]
    for i in range(0,iterations):
        sentences+=[normalise(sent) for sent in rcr.sample_sents(samplesize=samplesize)]
        print("Completed {}%".format(i))
    print("Completed 100%")
    return sentences

#uses context window of size 1 to generate a dictionary feature representation of all words in the corpus
def generate_features(sentences,window=1):
    mydict={}
    for sentence in sentences:
        for i,token in enumerate(sentence):
            current=mydict.get(token,{})
            features=sentence[max(0,i-window):i]+sentence[i+1:i+window+1]
            for feature in features:
                current[feature]=current.get(feature,0)+1
            mydict[token]=current
    return mydict


def semantic_similarity(word1, word2, speech, measure):
  """
  Finds the highest similarity score for the given pair of words. Goes through each combination of all senses.

  :param word1: First word in the pair of words
  :param word2: Second word in the pair of words
  :param speech: part of speech e.g. nw.NOUN
  :param measure: String representing the type of similarity measure ("path" = path ; "res" = Resnik  ;  "lin" = Lin)
  :return: The highest similarity score across all senses and all parts of speech
  """
  #error handling if invalid measure input is given
  if measure not in ["path","res","lin"]:
    raise ValueError("Not a valid similarity type \n Must be 'path'(path), 'res'(Resnik) or 'lin'(Lin)")

  greatest = 0
  conceptsA = wn.synsets(word1,speech)
  conceptsB = wn.synsets(word2,speech)
  #finds similarity score for every combination of senses
  for conceptA in conceptsA:
    for conceptB in conceptsB:
      if measure == "path":
        similarity = wn.path_similarity(conceptA,conceptB)
      elif measure == "res":
        similarity = wn.res_similarity(conceptA,conceptB,brown_ic)
      elif measure == "lin":
        similarity = wn.lin_similarity(conceptA,conceptB,brown_ic)
      if similarity == None : continue #error checking if similarity scorce not possible
      if similarity>greatest:
          greatest = similarity #if new highest similairty is found, set it to the greatest
  return greatest
