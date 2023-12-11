import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#initializam si antrenam modelul SVM
# SVC = suport vector clasification - masina de suport de a gasit hiperplanul care separa vectorul in clasele dorite
# se folosesc diferite tipuri de kernel
# alegerea kernelului e in functie de tipul de date
# kernel ='linear'
# kernel ='polinomial' introduce modele polinomiale
# kernel ='rbf' sau gaussian transforma datele intr-un spatiu ...
# kernel ='sigmoid' aduce nonlinearitate

# kernelul se alege prin incercare pana cand modelul face ce trebuie
# c - controleaza penalizarea erorilor, ajusteaza modelul
# se alege prin mai multe incercari, valoarea mai mare => penalaizari mai mici
# c mic => penalizare mai mare a erorilor de clasificare in timpul antrenarii
# c depinde de setul de date

# gamma - modalitate prin care controleaza cat de mult influenteaza un singur exemplu de antrenare
# gamma < mic => duce la o influenta mai mica asupra antrenarii
# gamma > mare => influenta mai mare asupra lucrurilor de antrenare

# random_state - parametru optional utilizat pentru a controla reproductibilitatea rezultatelor

model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42 )

model.fit(x_train, y_train)

predictions = model.predict(x_test)


acuratete = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
raport_clasificare = classification_report(y_test, predictions)

print(f"Acuratete: {acuratete}")
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Raport clasificare: \n{raport_clasificare}')



# Interpretarea rezultatelor
# avem 3 clase setosa, virginica, inca una
# de aceea in matrixea de confuzie avem 3x3
# acuratetea e de 100%
# scorul de 100% poate fi fals... nu exista test calsificat asa de corect
#
# la acuratete vedem 30 exemple care au fost clasificate corect - valoarea maxima pe care o poate avea testul de acuratete
# au fost realizate predictii corect opentru toate clasele
# diagonala principala arata nr de exmple clasificate corect si 10+9+11 =30 adica toate elementele
# celelelte valori sunt 0 deci nu au fost clasificate fals
# raportul de clasificare avem 1 deci totul e clasificat corect 100%
# avem un model robust care a facut clasificari foaret bune
# o performanta perfecta









