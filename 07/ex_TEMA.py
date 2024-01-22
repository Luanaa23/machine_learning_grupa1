import csv
import os
import gensim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# pasul 1 - incarcarea datelor
data = []
with open('Reviews.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0 # contor pentru numarul de linii citite
    for row in csv_reader: # fiecare rand din fisierul csv
        if line_count == 0:
            names = row  # prima linie (antetul) / ignor antetul
        else:
            data.append(row) # adaugă randul în lista data
        line_count += 1

# se separs textul recenziilor si etichetele sentimentelor
list_of_input = [data[i][0] for i in range(len(data))] # textele recenziilor
list_of_output = [data[i][1] for i in range(len(data))] # sentimentul recenziilor
label_names = list(set(list_of_output)) # lista de etichete de sentiment

# primele 2 recenzii și sentiment
print(list_of_input[:2])
print(label_names[:2])


# pasul 2 - impartire date in date de antrenament si test
np.random.seed(5)
no_of_samples = len(list_of_input)  #  Nr tot de recenzii
indexes = [i for i in range(no_of_samples)] # lista de indexi

# 80% din indexi pentru ANTRENAMENT - aleatoriu
#replace - are rolul sa asigure ca nu se alef aceeasi indexi de mai multe ori
trainSample = np.random.choice(indexes, int(0.8 * no_of_samples), replace=False)

# restul de 20% indexi pentru TESTARE
testSample = [i for i in indexes if not i in trainSample]

# separ inputurile si outputurile in seturi de ANTRENAMENT si TEST
training_input = [list_of_input[i] for i in trainSample]
training_output = [list_of_output[i] for i in trainSample]
test_inputs = [list_of_input[i] for i in testSample]
test_outputs = [list_of_output[i] for i in testSample]

print(training_input[:3]) # primele 3 recenzii de antrenament

# pasul 3: extragerea caracteristicilor

# # reprezentarea 1 - Bag of Words
# vectorizer = CountVectorizer()
# # transform textul intr o reprezentarea numerica bazata pe ponderile Bag of Words
# train_features = vectorizer.fit_transform(training_input)
# test_featrures = vectorizer.transform(test_inputs)

# #primele 10 cuvinte din vocabular si caracteristicile asociate
# print(f"vocabular reprezentare 1: {vectorizer.get_feature_names_out()[:10]}")
# print(f"features reprezentare 1: {train_features.toarray()[:3][:10]}")

# reprezentarea 2 - TF-IDF
vectorizer = TfidfVectorizer(max_features=50)
# transform textul intr o reprezentarea numerica bazata pe ponderile tfid
# max features = 50 - cele mai frecvente 50 cuvinte din reprezntarea nostra
training_features = vectorizer.fit_transform(training_input)
test_features = vectorizer.transform(test_inputs)


#primele 10 cuvinte din vocabular si caracteristicile asociate
print(f"vocabular reprezentare 2: {vectorizer.get_feature_names_out()[:10]}")
print(f"fearures reprezentare 2: {training_features.toarray()[:3]}")


# ## IMI BLOCHEAZA LAPTOPUL
# #reprezentarea 3 - Word2Vec
#
# crtDir = os.getcwd()
# modelPath = os.path.join(crtDir, 'GoogleNews-vectors-negative300.bin')
#
# word2vecModel300 = gensim.models.KeyedVectors.load_word2vec_format(modelPath, binary=True)
#
# print(word2vecModel300.most_similar('support'))
# print(f"vector for house: {word2vecModel300['house']}")



# # #reprezentarea 4 - N-grams
# # utilizez TfidfVectorizer pentru a extrage bigrame si trigrame
# # 'ngram_range' - parametrii care pot fi ajustati pentru a selecta diferite dimensiuni ale N-gramelor
# # max_features=100 limitează numărul de caracteristici la cele mai frecvente 100 de N-grame
# ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3), max_features=100)
#
# # transform textele de antrenament si test
# training_features_ngram = ngram_vectorizer.fit_transform(training_input)
# test_features_ngram = ngram_vectorizer.transform(test_inputs)
#
# # primele 10 caracteristici (N-grame) din vocabular
# print(f"vocabular N-grams reprezentare 4: {ngram_vectorizer.get_feature_names_out()[:10]}")
# print(f"features N-grams reprezentare 4: {training_features_ngram.toarray()[:3][:10]}")



# pasul 4 - antrenare model de invatare SUPERVIZATA - regresie logistica
# aleg caracteristicile pentru antrenare si testare - aici folosesc TF-IDF

# # #Bag of Words
# X_train = train_features
# X_test = test_featrures


# TF-IDF
X_train = training_features
X_test = test_features


# # N-grams
# X_train = training_features_ngram
# X_test = test_features_ngram

# convertesc etichetele text in valori numerice
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(training_output) # convert etichetele de antrenament
y_test = label_encoder.transform(test_outputs) # convert etichetele de testare

# pasul 4.1 - se antreneaza modelul de regresie logistică
model = LogisticRegression() # modelul de regresie logistica
model.fit(X_train, y_train) # antren modelul pe setul de antrenament

# pasul 4.2 - testare model regresie logisica
predictions = model.predict(X_test) # predictii pe setul de test

# pasul 4.3 - calcul metrici de performanta
acuratetea = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
raport_clasificare = classification_report(y_test, predictions)

print(f"Acuratete: {acuratetea}")
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Raport clasificare: \n{raport_clasificare}')



# pasul 5 - antrenare model de invatare NESUPERVIZATA (clustering)

# NUU !!! - IMI BLOCHEAZA LAPTOPUL
# word2vecModel300
# def feature_calcul(model, data):
#     features = []
#     phrases = [phrase.split() for phrase in data]
#     for phrase in phrases:
#         vectors = [model[word] for word in phrase if (len(word) > 2) and (word in model.key_to_index)]
#         if len(vectors) == 0:
#             result = [0.0] * model.vector_size
#         else:
#             result = np.sum(vectors, axis=0) / len(vectors)
#         features.append(result)
#     return features
#
# training_features = feature_calcul(word2vecModel300, training_input)
# test_features = feature_calcul(word2vecModel300, test_inputs)


# #Bag of Words
# unsupervisedClassifier = KMeans(n_clusters=2, random_state=0, n_init=10) # modelul pt 2 clustere
# unsupervisedClassifier.fit(train_features) # antr modelul K-means pe setul de antrenament


# #TF-IDF
# antrenez modelul K-means folosind caracteristicile TF-IDF
unsupervisedClassifier = KMeans(n_clusters=2, random_state=0, n_init=10) # modelul pt 2 clustere
unsupervisedClassifier.fit(training_features) # antr modelul K-means pe setul de antrenament


# # #N-grams
# unsupervisedClassifier = KMeans(n_clusters=2, random_state=0, n_init=10) # modelul pt 2 clustere
# unsupervisedClassifier.fit(training_features_ngram) # antr modelul K-means pe setul de antrenament



# # pasul 6 - testare model K-means

# #Bag of Words
# computedTestIndexes = unsupervisedClassifier.predict(test_featrures) # predictii pe setul de test
# computedTestOutputs = [label_names[value] for value in computedTestIndexes] # conv indexii clusterelor in etichete


# #TF-IDF
computedTestIndexes = unsupervisedClassifier.predict(test_features) # predictii pe setul de test
computedTestOutputs = [label_names[value] for value in computedTestIndexes] # conv indexii clusterelor in etichete

# # #N-grams
# computedTestIndexes = unsupervisedClassifier.predict(test_features_ngram) # predictii pe setul de test
# computedTestOutputs = [label_names[value] for value in computedTestIndexes] # conv indexii clusterelor in etichete


# pasul 6 - calcul metrica de performanta
print(f'acc: {accuracy_score(test_outputs, computedTestOutputs)}')




# Rezultate
# ['The rooms are extremely small, practically only a bed.', 'Room safe did not work.']
# primele două exemple de recenzii.

# ['negative', 'positive']: datele sunt clasificate in doua categorii

# ['still easy to reach.', 'Some thinks didnt work well : air, tv , open windows,', 'Room was not cleaned even once during our stay.']
# primele trei recenzii din setul de date de antrenament

# vocabular reprezentare 2: ['all' 'and' 'are' 'as' 'bathroom' 'bed' 'clean' 'cold' 'comfortable' 'could']
# primele 10 cuvinte din vocabularul creat de TF-IDF
# cuvinte comune

# fearures reprezentare 2:
# [[0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.         0.         1.         0.         0.
#   0.         0.         0.         0.         0.         0.
#   0.         0.        ] ..................
# matricei TF-IDF pentru primele trei recenzii
# frecvența relativă a cuvintelor din vocabularul TF-IDF
# fiecare rând reprezintă vectorul de caracteristici TF-IDF pentru o recenzie
# valoarea 1 - prezența cuvantului din vocabular in recenzia respectiva
# valoarea 0 - absenta cuvantului din vocabular in recenzia respectiva


# Acuratete: 0.6428571428571429
# acuratețea modelului de inv SUPERVIZATA
# mssura a proporției de predicții corecte (atât pozitive, cat si negative)
# aproximativ 64% din toate exemplele au fost clasificate corect


# Confusion Matrix:
# [[24  1]  # 25 de recenzii reale negative
#  [14  3]] # 17 de recenzii reale pozitive
# 25 de recenzii reale negative
# din care 24 au fost prezise corect ca fiind negative, și 1 a fost prezis greșit ca pozitivă
#
# 17 recenzii reale pozitive
# doar 3 au fost prezise corect ca fiind pozitive
# iar 14 au fost prezise greșit ca fiind negative
#
# Acest lucru indică o tendință a modelului de a clasifica recenziile ca fiind negative



# Raport clasificare:
#               precision    recall  f1-score   support
#
#            0       0.63      0.96      0.76        25
#            1       0.75      0.18      0.29        17
#
#     accuracy                           0.64        42
#    macro avg       0.69      0.57      0.52        42
# weighted avg       0.68      0.64      0.57        42

# precizia (procentul de predicții corecte pentru o clasa)
# clasa 0 este 0.63, ceea ce înseamnă că 63% din cazuri au fost corect clasificate ca negative
# din totalul exemplelor identificate ca negative
# clasa 1 este 0.75, ceea ce înseamnă că 75% din cazuri au fost corect clasificate ca pozitive
# din totalul exemplelor identificate ca pozitive

# recall (procentul de cazuri reale pentru o clasa care au fost prezise corect)
# clasa 0 este 0.96, ceea ce înseamnă că
# 96%  din exemplele reale de negative au fost identificate corect
# clasa 1 este 0.18, ceea ce înseamnă că doar
# 18% din exemplele reale de pozitive au fost identificate corect

# scorul F1 (media armonica intre precizie si recuperare) pentru fiecare class - valoare mai mare indica o performanta mai buna
# clasa 0 este 0.76 - o valoare mai mare indica o performanța mai buna
# clasa 1 este 0.29 - o valoare mai scazuta indica o performanța mai slaba


# Support arată numărul de exemple reale din fiecare clasa

# Deoarece nu au fost identificate corect exemplele de pozitive (doar 3) modelul are o problemă semnificativă.



# Măsuri Macro și Weighted Average:
# - Macro avg: Media neponderată a măsurilor pentru fiecare clasă. Nu ține cont de dezechilibrul între clase.
# - Weighted avg: Media ponderată a măsurilor pentru fiecare clasă, ponderată după suportul fiecărei clase. Ponderarea este
# importantă în cazul unui dezechilibru între clase.




# acc: 0.6666666666666666
# acuratețea modelului de inv NESUPERVIZATA
# mssura a proporției de predicții corecte (atât pozitive, cat si negative)
# aproximativ 66% din toate exemplele au fost clasificate corect










