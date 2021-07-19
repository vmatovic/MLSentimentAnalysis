
import nltk
# nltk.download()

"""# Neke biblioteke"""

import os
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
import numpy as np
import random

"""# Ucitavanje"""
print('Ucitavanje')
comm_all = []

def read_file(s):
  with open(s, 'r', encoding="utf8") as f:
    s = f.read()
    f.close()
    return s

def all_comments(pth):
	ls = []
	os.chdir(pth)
	for file in os.listdir():
		if file.endswith('.txt'):
			f_path = file
			ls.append(read_file(f_path))
			comm_all.append(read_file(f_path))
  
	return ls


pos_comments = []
neg_comments = []
path_pos = 'data/imdb/pos'
path_neg = 'data/imdb/neg'



pos_comments = all_comments(path_pos)
os.chdir('..')
os.chdir('..')
os.chdir('..')
neg_comments = all_comments(path_neg)
random.shuffle(pos_comments)
random.shuffle(neg_comments)

print('Ciscenje reci')
"""# Ciscenje reci"""

stops = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(ls):
  global stops
  for i, s in enumerate(ls):
    tmp = []
    t1 = []
    tmp = word_tokenize(s)
    tmp = list(map(lambda x: x.lower(), tmp))
    tmp = [s for s in tmp if s.isalpha() and len(s)>1 and not s in stops and s != 'br']
    tmp = pos_tag(tmp)
    for s in tmp:
      if s[1].startswith('NN'):
        t1.append(lemmatizer.lemmatize(s[0], pos='n'))
      elif s[1].startswith('VB'):
        t1.append(lemmatizer.lemmatize(s[0], pos='v'))
      elif s[1].startswith('JJ'):
        t1.append(lemmatizer.lemmatize(s[0], pos='a'))
    #tmp = [s[0] for s in tmp if s[1].startswith('NN') or s[1].startswith('JJ') or s[1].startswith('VB')]
    tmp = t1[:]
    ls[i] = tmp
  
  return ls


pos_comments = clean_text(pos_comments)
neg_comments = clean_text(neg_comments)
print('Stemming')
"""# Stemming"""

porter = PorterStemmer()


def stem_it(ls):
  global porter

  for i, s in enumerate(ls):
    tmp = []
    for s1 in s:
      tmp.append(porter.stem(s1))
    ls[i] = tmp  
  return ls

pos_comments = stem_it(pos_comments)
neg_comments = stem_it(neg_comments)


big_list = []
for pc in pos_comments:
  big_list += pc
for nc in neg_comments:
  big_list += nc
freq = FreqDist(big_list)
big_list = freq.most_common(10000)
big_list = [bb[0] for bb in big_list]
print('Bag of Words')
"""# Bag of Words"""

vocab_set = set()

def make_bow(ls):
  global vocab_set
  for word in ls:
    vocab_set.add(word)

make_bow(big_list)

vocab = list(vocab_set)

"""# Neki sinonimi"""

syn_lems = []


def make_dict(wrd, a=0, b=-1):
  global syn_lems
  synsets = wordnet.synsets(wrd)[a:b]
  
  for syn in synsets:
    syn_lems += syn.lemmas()


make_dict('good')
make_dict('like')
make_dict('love', 6, 9)
make_dict('great')
pos_syn_lems = [porter.stem(sy.name()) for sy in syn_lems]
pos_syn_lems = set(pos_syn_lems)

syn_lems = []
make_dict('bad')
make_dict('hate')
make_dict('awful', 0, 5)
make_dict('avoid')
neg_syn_lems = [porter.stem(sy.name()) for sy in syn_lems]
neg_syn_lems = set(neg_syn_lems)
print('Trening')
"""# Trening data"""

X = np.zeros((2, len(vocab)), dtype=np.int64)

for di in range(1000):
  d = pos_comments[di]
  for wi in range(len(vocab)):
    word = vocab[wi]
    cnt = d.count(word)
    X[0][wi] += cnt

pc = len(pos_comments)
for di in range(1000):
  d = neg_comments[di]
  for wi in range(len(vocab)):
    word = vocab[wi]
    cnt = d.count(word)
    X[1][wi] += cnt


"""# Frekvencija"""

alpha = 2
X1 = np.zeros((2, len(vocab)), dtype=np.float64)

for i in range(2):
  for w in range(len(vocab)):
    inflation = 0
    if i == 0 and vocab[w] in pos_syn_lems:
      inflation = 0.001
    elif i == 1 and vocab[w] in neg_syn_lems:
      inflation = 0.001
    br = X[i][w] + alpha
    im = np.sum(X[i]) + len(vocab)*alpha
    X1[i][w] = br/im + inflation
print('Predvidjanje')
"""# Predvidjanje"""

br_pos_p = 0
br_neg_p = 0
br_pos_n = 0
br_neg_n = 0

for di in range(1000, 1250):
  tmp = np.zeros(len(vocab))
  d = pos_comments[di]
  for wi in range(len(vocab)):
    word = vocab[wi]
    cnt = d.count(word)
    tmp[wi] += cnt
  
  prob_positive = 0.5
  prob_negative = 0.5

  for wi in range(len(vocab)):
    cnt = tmp[wi]
    prob_positive += np.log(X1[0][wi]) * cnt
    prob_negative += np.log(X1[1][wi]) * cnt
  
  if prob_positive > prob_negative:
    br_pos_p += 1
  else:
    br_neg_p += 1

for di in range(1000, 1250):
  tmp = np.zeros(len(vocab))
  d = neg_comments[di]
  for wi in range(len(vocab)):
    word = vocab[wi]
    cnt = d.count(word)
    tmp[wi] += cnt
  
  prob_positive = 0.5
  prob_negative = 0.5

  for wi in range(len(vocab)):
    cnt = tmp[wi]
    prob_positive += np.log(X1[0][wi]) * cnt
    prob_negative += np.log(X1[1][wi]) * cnt
  
  if prob_positive > prob_negative:
    br_pos_n += 1
  else:
    br_neg_n += 1

print('Broj stvarnih positivnih:', br_pos_p)
print('Broj laznih negativnih:', br_neg_p)
print('Broj laznih pozitivnih:', br_pos_n)
print('Broj stvarnih negativnih:', br_neg_n)

"""# Matrica konfuzije"""

mat_conf = [[br_neg_n, br_pos_n], [br_neg_p, br_pos_p]]
print(mat_conf)

# primer: [[198, 52], [50, 200]]


pc_inone = []
nc_inone = []
for pc in pos_comments:
  pc_inone += pc
for nc in neg_comments:
  nc_inone += nc
frq_pos = FreqDist(pc_inone)
frq_neg = FreqDist(nc_inone)
popular_pos_words = frq_pos.most_common(5)
popular_neg_words = frq_neg.most_common(5)
print(popular_pos_words)
print(popular_neg_words)

popular_pos_words = frq_pos.most_common(2000)
popular_neg_words = frq_neg.most_common(2000)
ppw_dict = {}
pnw_dict = {}
for ppw in popular_pos_words:
  ppw_dict[ppw[0]] = ppw[1]

for pnw in popular_neg_words:
  pnw_dict[pnw[0]] = pnw[1]

lr_word = []
for ind, v in ppw_dict.items():
  if ind in pnw_dict.keys():
    lr_word.append((ind, v/pnw_dict[ind]))

lr_word = sorted(lr_word, key=lambda x: x[1], reverse=True)
print(lr_word[:5])
print(lr_word[-5:])

# Najpopularije reci iz pozitivnih komentara: [('film', 2429), ('movi', 2309), ('make', 1127), ('see', 1053), ('good', 822)]
# Najpopularije reci iz negativnih komentara: [('movi', 2865), ('film', 2043), ('make', 1178), ('see', 1070), ('get', 1008)]

# LR(rec):

# [('superb', 5.222), ('affect', 4.555), ('harri', 4.5384), ('henri', 4.5), ('rise', 4.4615)]
# [('ridicul', 0.153), ('aw', 0.148), ('terribl', 0.1417), ('wast', 0.1296), ('stupid', 0.098)]

# Vidimo da LR bolje reprezentuje koje reci vise prilice pozitivnim/negativnim komentarima nego njihove najpopularnije reci