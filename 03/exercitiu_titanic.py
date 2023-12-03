import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic_data = sns.load_dataset('titanic')
print(titanic_data.to_string())

sns.countplot(x='class', hue='survived', data=titanic_data)
plt.title('Analiza supavietuire in functie de clasa')
plt.show()

sns.histplot(x='age', hue='survived', data=titanic_data, kde=True)
plt.title('Analiza supravietuire in functie de varsta')
plt.show()

sns.countplot(x='sex', hue='survived', data=titanic_data)
plt.title('Analiza supravietuire in functie de sex')
plt.show()

sns.countplot(x='embarked', hue='survived', data=titanic_data)
plt.title('Analiza supravietuire in functie Portul de imbarcare')
plt.legend()
plt.show()
