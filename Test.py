import pandas as pd
import numpy as np
import re

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE

#======Hyperparams======
WINDOW_SIZE = 2
EMBEDDING_DIM = 73

def prep_str(str):
    # sets
    stop_words_rus = set(stopwords.words('russian'))
    stop_words_eng = set(stopwords.words('english'))
    abbreviation = ["рад", "мрад", "мкрад",  # угол
                    "см", "м", "мм", "мкм", "нм", "дм",  # метр
                    "кг", "мг", "мкг", "г", "т",  # вес
                    "мин", "ч", "сут", "с", "мс", "мкс", "нс",  # время
                    "л",  # объем
                    "гц", "ггц", "мгц", "кгц",  # Гц
                    "шт",  # кол-во
                    "ом", "а", "в",  # эл-тех
                    "млн", "тыс", "млрд", "руб", "ме",  # денеж.ср.
                    "бит", "байт", "кбай", "мбайт", "гбайт", "тбайт", "мбайт"]  # информ.

    str = re.sub(r'\d+[xX]\d+| \d+[xX]\d+[xX]\d+', '', str)  # eng
    str = re.sub(r'\d+[хХ]\d+| \d+[хХ]\d+[хХ]\d+', '', str)  # rus
    str = re.sub(r'[A-z]', "", str)  # удаление английских литералов
    str = re.sub(r'\d+', '', str).lower()
    str = re.sub("[^\w]", " ", str).split()

    words = [word for word in str if
             (not word in abbreviation) and (not word in stop_words_rus) and (not word in stop_words_eng) and (
                     len(word) > 2)]
    # print(words)

    str = ' '.join(words)
    #str = noun_selector_for_str(str)
    return str

def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)#создаем вектор в длинну словаря из нулей
    temp[data_point_index] = 1 #создаем one hot encoding vector
    return temp

#======InputData======
input_str = "Бывший главный тренер сборной Финляндии по лыжным гонкам Кари-Пекка Кюре негативно высказался о норвежских лыжниках, принимающих противоастматические препараты. Его слова приводит Ilta-Sanomat."\
            "Кюре остался недоволен словами норвежки Астрид Якобсен о том, что календарь Кубка мира очень плотный, и большие нагрузки приводят к появлению астмы. «Это ложь и попытки оправдаться. Нагрузки на дыхательную систему снизились по сравнению с предыдущими годами», — заявил он."\
            "По мнению финна, в 70-е и 80-е годы прошлого века Кубок мира длился дольше, а погодные условия были жестче. «Пусть норвежские астматики не участвуют в Кубке мира. Спорту не нужны больные атлеты. Пусть организуют свой Кубок астматиков и выступают там с другими астматиками», — добавил он."

#======PrepareData======
pure_str = prep_str(input_str)
print(pure_str)

words = set(pure_str.split())
vocab_size = len(words)
print(vocab_size)

word2int = {}
int2word = {}
for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

sentences = []
raw_sentences = input_str.split('.')

for sentence in raw_sentences:
    sentence = prep_str(sentence)
    sentences.append(sentence.split())
print(sentences)

data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])
print(data)

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))#[0.]
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


#======TensorFlow======





'''
x = tf.placeholder(tf.float32, shape=(None, vocab_size))        #vocab_size=?x73
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))  #vocab_size=?x73


#весовая матрица 7хEMBEDDING_DIM состоит из рандомных компонентов
W1 = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_DIM],-1.0,1.0))
#свободный член размерностью EMBEDDING_DIM состоит из рандомных компонентов
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))


#задаем параметры линейного hidden layer W1*x+b
hidden_representation = tf.add(tf.matmul(x,W1), b1)
#hidden_representation выход с hidden layer

#инициализируем еще 2 параметра w2 and b2, проделываем преобразования w2*hidden_representation+b2 -> softmax
W2 = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_DIM],-1.0,1.0))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))#вероятности
#input_one_hot  --->  embedded repr. ---> predicted_neighbour_prob
'''













'''



sentences = []

bb=[]
words = []
sentences = input_str.split(".")
for sentence in sentences:
    bb = [prep_str(sentence)]
    pure_str.append(bb)

    words.append(sentence.split())

print(pure_str)
words = set (words)
print(words)
word2int = {}
int2word = {}
words = []


for word in input_str.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)

#words = pure_str.split()
#print(words)
#print(len(words))
#print(set(words))
vocab_size = len(set(words))

for i,word in enumerate(words):
    word2int[word]=i
    int2word[i] = word

data = []
for elem in pure_str:
    for word_index, word in enumerate(elem):
        for nb_word in elem[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])#[[local1,neighbor1],[local1,neighbor2],[]]
print(data)



'''

