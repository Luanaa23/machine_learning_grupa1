# Temă Sesiunea 2. Analiza si vizualizare de date Analiza Setului de Date Airbnb:
# Utilizați un set de date Airbnb pentru a analiza prețurile și ocuparea locuințelor într-o anumită locație.
# Creați hărți de căldură pentru a evidenția zonele populare și grafice pentru a înțelege factorii care influențează prețurile.
# Setul de date trebuie sa aiba urmatoarea structura:
# data =
# {'Neighborhood': ['A', 'B', 'C', 'D', 'E'],
# 'Price': [100, 120, 80, 150, 90],
# 'Occupancy': [80, 60, 90, 50, 70],
# 'Review_Score': [4.5, 4.2, 4.8, 3.9, 4.6]}
# Creati un set de date cu 100 de inregistrari pentru structura de mai sus.
# In coloana 'Neighborhood' trebuie sa avem date duplicat.

import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# # Generare 100 numere pentru Neighborhood
# lista_Neighborhood = ['A', 'B', 'C', 'D', 'E']
# Neighborhood =  random.choices(lista_Neighborhood, k=100)
# print(Neighborhood)


# # Generare 100 numere pentru Price
# Price = []
# for i in range(0,100):
#   n = random.randint(80,150)
#   Price.append(n)
# print(Price)
#
# Price = [random.randint(80, 150) for i in range(0,100)]
# print(Price)


# # Generare 100 numere pentru Occupancy
# Occupancy = [random.randint(50, 90) for i in range(0,100)]
# print(Occupancy)


# # Generare 100 numere pentru Review_Score
# Review_Score = [round(random.uniform(3, 5),1) for i in range(0,100)]
# print(Review_Score)


lista_Neighborhood = ['A', 'B', 'C', 'D', 'E']
data = {'Neighborhood': random.choices(lista_Neighborhood, k=100),
        'Price': [random.randint(80, 150) for i in range(0,100)],
        'Occupancy': [random.randint(50, 90) for i in range(0,100)],
        'Review_Score': [round(random.uniform(3, 5),1) for i in range(0,100)]}

df = pd.DataFrame(data)

print(df.to_string())

#hărți de căldură pentru a evidenția zonele populare

# Funcția df.corr() din pandas este utilizată pentru a calcula coeficientul de corelație între coloane,
# dar aceasta funcționează doar pe date numerice.
# coloana Neighborhood este compusă din șiruri de caractere ('A', 'B', 'C', 'D', 'E')
# pentru a putea realiza harta de caldura se va exclude coloana "Neighborhood""

# Se selecteaza coloanele care contin date numerice (fara coloane 'Neighborhood')
df_filtrat = df[['Price', 'Occupancy', 'Review_Score']]

# se deseneaza harta de căldură folosind matricea de corelație
plt.figure(figsize=(10,8))
sns.heatmap(df_filtrat.corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de corelatie')
plt.show()


# grafice pentru a înțelege factorii care influențează prețurile

#Analiza Prețurilor în Funcție de Vecinătate: cu barplot
plt.figure(figsize=(10,6))
sns.barplot(x='Neighborhood', y='Price',hue='Neighborhood', data=df)
plt.title('Distribuția prețurilor pe cartiere')
plt.show()


#Analiza Prețurilor în Funcție de Vecinătate: cu violinplot
plt.figure(figsize=(14,6))
sns.violinplot(x='Neighborhood', y='Price', hue='Neighborhood', palette='Dark2', data=df, legend=False)
plt.title('Distribuția prețurilor pe cartiere')
plt.show()


#Relația dintre Scorul de Review și Preț: cu scatterplot
plt.figure(figsize=(15,8))
sns.scatterplot(x='Review_Score', y='Price', hue='Neighborhood', palette='Dark2', data=df)
plt.title('Relația dintre scorul de review și preț')
plt.xlabel('Scorul de Review')
plt.ylabel('Preț')
plt.show()


#Relatia dintre Ocupare si Preț: cu scatterplot
plt.figure(figsize=(15,8))
sns.scatterplot(x='Occupancy', y='Price', hue='Neighborhood',palette='Dark2', data=df)
plt.title('Relația dintre ocupare și preț')
plt.xlabel('Ocupare')
plt.ylabel('Preț')
plt.show()


#Pair plot pentru setul de date Airbnb
sns.pairplot(df, hue='Neighborhood', palette='husl')
plt.suptitle('Pair plot pentru setul de date Airbnb', y=1.02)
plt.show()
