import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE

#Предобработка
#входная строка
corpus_raw = 'Мое дело целиком лежит на моих плечах . Две барышни с пишущими машинками и конторскими книгами в передней, моя комната с письменным столом, денежным ящиком, столом для совещаний, мягким креслом и телефоном – вот весь мой аппарат . Его так легко обозреть, им так легко управлять . Я совсем молод, и дела у меня сами идут . Я не жалуюсь, я не жалуюсь . С нового года один молодой человек без раздумий снял пустующую соседнюю квартирку, со съемом которой я, растяпа, так долго медлил . Тоже комната с передней, но, кроме того, и кухня . Комната и передняя мне не помешали бы, обе мои барышни иногда уже чувствовали чрезмерную нагрузку, – но на что мне нужна была кухня? Из-за этой закавычки я и упустил квартиру . Теперь там расположился этот молодой человек . Гаррас его фамилия . На двери табличка: «Гаррас, контора» . Я навел справки, мне сказали , что это дело подобное моему . От предоставления ему кредита не то чтобы предостерегали, ведь речь шла о молодом, растущем человеке, у которого, возможно, есть будущее, однако не то чтобы и советовали предоставлять ему кредит, ибо в данный момент состояния, судя по всему, нет . '

'''
Ах, эти убого тонкие стены, предающие человека, честно трудящегося, а нечестного укрывающие. Мой телефон висит на стене, которая отделяет меня от соседа. Однако я отмечаю это лишь как особенно иронический факт. Даже если бы он висел на противоположной стене, в соседней квартире было бы все слышно. Я отучился называть по телефону имена клиентов. Но не требуется, разумеется, большой хитрости, чтобы угадывать эти имена по характерным, но неизбежным поворотам разговора... Иногда я от беспокойства пляшу на цыпочках с наушником вокруг аппарата и все-таки не могу предотвратить разглашения тайн.
Конечно, из-за этого мои деловые решения становятся неуверенными, мой голос нетвердым. Что делает Гаррас, когда я говорю по телефону? Если бы я захотел сильно преувеличить – а это часто приходится делать, чтобы обрести ясность, – я мог бы сказать: Гаррасу телефон не нужен, он пользуется моим, он придвинул к стенке свой диванчик и слушает, а я, когда раздается звонок, должен бежать к телефону, выслушивать желания клиента, принимать важные решения, истово уговаривать – но тем самым прежде всего поневоле давать отчет Гаррасу через стенку.
Может быть, он даже не дожидается конца разговора, а поднимается после тех слов, которые достаточно прояснили ему дело, мечется по своему обыкновению по городу и, прежде чем я по.вешу трубку, уже, может быть, начинает действовать против меня.'
'''
#corpus_raw = open('data/test_text.txt', 'r')

# convert to lower case
corpus_raw = corpus_raw.lower()

words = []
for word in corpus_raw.split():
    if word != '.': # because we don't want to treat . as a word
        words.append(word)

words = set(words)# so that all duplicate words are removed


word2int = {}
int2word = {}
vocab_size = len(words) # gives the total number of unique words

#создаем 2 словаря{Позиция:Слово} и {Слово:Позиция}
for i,word in enumerate(words):
    word2int[word] = i
    #print (word2int)
    int2word[i] = word

'''
# реализовано для быстрого поиска
print(word2int['queen'])
-> 42 (say)

print(int2word[42])
-> 'queen'
'''
#print(word2int['king'])
#print(word2int['queen'])
#формируем из единой строки список токенов
# raw sentences is a list of sentences.
raw_sentences = corpus_raw.split('.')
sentences = []
for sentence in raw_sentences:
    print(sentence)
    sentences.append(sentence.split())
print(sentences)  # [[1st_word_1st_sen,2nd_word_1st_sen,...],[1st_word_2nd_sen,2nd_word_2nd_sen,...],[]]

#гипперпараметр
WINDOW_SIZE = 2
#алгоритм при помощи словарей создает список [[label0,neighbor0], ...]
data = []
for sentence in sentences:
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0) : min(word_index + WINDOW_SIZE, len(sentence)) + 1] :
            if nb_word != word:
                data.append([word, nb_word])

print (data)

#На данном этапе нами сформированы данные для обучения
#==============================================

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)#создаем вектор в длинну словаря из нулей
    temp[data_point_index] = 1 #создаем one hot encoding vector
    return temp

x_train = [] # input word
y_train = [] # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[ data_word[0] ], vocab_size))
    y_train.append(to_one_hot(word2int[ data_word[1] ], vocab_size))



# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
#print (x_train, y_train)


#Make the tensorflow model

# making placeholders for x_train and y_train
x = tf.placeholder(tf.float32, shape=(None, vocab_size))#vocab_size=?x7
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))#vocab_size=?x7

#на данном этапе мы реализуем skipgram помещая в х и y_label получившиеся вектора

#размерность внутреннего слоя hidden layer
#embedding_dim количество входов внутреннего слоя
EMBEDDING_DIM = 64 # you can choose your own number

#весовая матрица 7хEMBEDDING_DIM состоит из рандомных компонентов
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
#свободный член размерностью EMBEDDING_DIM состоит из рандомных компонентов
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
#задаем параметры линейного hidden layer W1*x+b
hidden_representation = tf.add(tf.matmul(x,W1), b1)
#hidden_representation выход с hidden layer

#инициализируем еще 2 параметра w2 and b2, проделываем преобразования w2*hidden_representation+b2 -> softmax
W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))#вероятности
#input_one_hot  --->  embedded repr. ---> predicted_neighbour_prob


#Процесс обучения
sess = tf.Session()#старт сессии

init = tf.global_variables_initializer()#инициализация переменных
sess.run(init) #make sure you do this!

#переменные проинициализированны

# define the loss function:
#ошибка при прогоне данных с использованием кросс-энтропии

#считаем loss
        #??????почему мы умножаем на y_label 1)попробовать убрать
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))

'''
параметры команд
tf.math.reduce_mean(#вычисляет среднее значение по элементам
    input_tensor,#
    axis=None,#
    keepdims=None,#
    name=None,#имя оператора, опционально
    
    reduction_indices=None,#
    keep_dims=None#
)

tf.math.reduce_sum(#сумма элементов матрицы
    input_tensor,#матрица элементы которой будут сложены 
    axis=None,0-по столбцам 1-по строкам
    keepdims=None,#помещать данные в отдельные списки
    name=None,#имя оператора, опционально
    
    reduction_indices=None,#тоже что и axis 
    keep_dims=None# то же что и keepdims
)

tf.math.log(
    x,#вписываем имя тензора одного из типов, типы мы указывали выше
    name=None #имя оператора, опционально
)

'''


# define the training step:
#инициализация градиентного спуска, минимизируется cross_entropy_loss(отличается от loss из материалов Козлова)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)

# train for n_iter iterations
n_iters = 10000 #число прогонов, повторений (итераций)


for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))#1.3208

vectors = sess.run(W1+b1)
#print(vector)

#dirty implementation :)
def euclidean_dist(vec1, vec2):#расчет евклидового расстояния между векторами
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):#поиск ближайшего вектора
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index #возвращает индекс ближайшего к выбранному вектора
#значения меняются в связи с тем, что мы используем множество words внутри которого при перезапуске меняется порядок
#corpus_raw = 'He is the king . The king is royal . She is the royal  queen '
#print(int2word[find_closest(word2int['king'], vectors)])#she#берем номер слова 'king' в множестве и сумму 2х матриц??
#print(int2word[find_closest(word2int['queen'], vectors)])#is
#print(int2word[find_closest(word2int['royal'], vectors)])#he

'''
нужно понять подходит ли этот код только для представления в 2д пространстве или это можно исправить
визуализацию сделать
'''


from sklearn.manifold import TSNE
from sklearn import preprocessing
#реализация
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
model = TSNE(n_components=2, random_state=0)#n_components задает размерность пространства, max (n=3)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)


#нормализация векторов
normalizer = preprocessing.Normalizer()
vectors = normalizer.fit_transform(vectors, 'l2')#l1 манхетеновское расстояние, l2 евклидово растояние

#print(vectors)


import matplotlib.pyplot as plt


fig, ax = plt.subplots()
print(words)
for word in words:
    print(word, vectors[word2int[word]])
    ax.annotate(word, (vectors[word2int[word]][0],vectors[word2int[word]][1] ))

ax.set_xlim(min([vectors[word2int[w]][0] for w in words])-1, max([vectors[word2int[w]][0] for w in words])+1)
ax.set_ylim(min([vectors[word2int[w]][1] for w in words])-1, max([vectors[word2int[w]][1] for w in words])+1)
plt.show()


import tensorflow as tf
x = tf.Variable([0.07, 2.0])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
v = sess.run(x)    
print(v) # will show you your variable.
print(int2word[find_closest(word2int['мне'], vectors)])
'''
http://qaru.site/questions/195890/tensorflow-get-current-value-of-a-variable
https://dev-ops-notes.ru/machine-learning/tensorfolow-%D0%BF%D0%B5%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5/
'''