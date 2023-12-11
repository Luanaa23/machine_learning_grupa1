from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('dataset.csv')

label_encoder = LabelEncoder()
data['locatie'] = label_encoder.fit_transform(data['locatie'])

x_train, x_test, y_train, y_test = train_test_split (
    data[['numar_camere', 'suprafata_utila', 'locatie', 'an_constructie', 'pret']],
    data['target'],
    test_size=0.2,
    random_state=42
)
# test_size = 20 % vor fi alocate testului de testare si restul antrenarii

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

prediction = model.predict(x_test)

acuratete = accuracy_score(y_test, prediction)
conf_matrix = confusion_matrix(y_test, prediction)
raport_clasificare = classification_report(y_test, prediction)

print(f"Acuratete: {acuratete}")
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'Raport clasificare: \n{raport_clasificare}')


# toate datele au fost clasificate corect
# precizia e de 100% - toate clsificate corect
# recall 100% - modelul a identificat corect toate apartementele si toate casele
# f1 a identificat tot 100%


