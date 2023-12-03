from faker import Faker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

fake = Faker()

data = {'Data': [], 'Produs': [], 'Cantitate': [], 'Pret_unitar': []}

for _ in range(100):
    data['Data'].append(fake.date_between(start_date='-30d', end_date= 'today'))
    data['Produs'].append(random.choice(['Produs_A', 'Produs_B', 'Produs_C']))
    data['Cantitate'].append(random.randint(10,50))
    data['Pret_unitar'].append(round(random.uniform(20,100),2))
df = pd.DataFrame(data)
df['Data'] = pd.to_datetime(df['Data'])
df.to_csv('vanzari.csv', index=False)

value = pd.read_csv('vanzari.csv')
sf = pd.DataFrame(value)

df['Venituri'] = df['Cantitate'] * df['Pret_unitar']

total_vanzari_zilnice = df.groupby('Data')['Venituri'].sum().reset_index()
plt.figure(figsize=(12,6))
sns.lineplot(x='Data', y='Venituri', data=total_vanzari_zilnice)
plt.title('Evolutie vanzari zilnice')
plt.xlabel('Data')
plt.ylabel('Venituri')
plt.show()

plt.figure(figsize=(10,6))
sns.histplot(df['Pret_unitar'], bins=20, kde=True, color='skyblue')
plt.title('Distributie preturi produs')
plt.xlabel('Pret unitar')
plt.ylabel('Numar produs')
plt.show()

plt.figure(figsize=(10,6))
sns.scatterplot(x='Pret_unitar', y='Cantitate', data=df, hue='Produs', palette='viridis')
plt.title('Relatia dintre pret si cantitatea vanduta')
plt.xlabel('Pret unitar')
plt.ylabel('Cantitate vanzare')
plt.legend(title='Produsul')
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de corelatie')
plt.show()



