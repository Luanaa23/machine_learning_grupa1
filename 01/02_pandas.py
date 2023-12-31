import pandas as pd

# data = {'Nume':['Ana', 'Bogdan', 'Cristina'],
#         'Varsta': [25,30,22],
#         'Salariu': [50000, 60000, 45000]}
# df = pd.DataFrame(data)
# print(df)
# nume = df['Nume']
# print(nume)
#
# salariu_bogdan = df.at[1, 'Salariu']
# print(salariu_bogdan)
#
# #adaugare coloana
# df['Experienta'] = [2,8,1]
# print(df)
# print('**************')
#
# #modificare coloane si linii
# df.set_index('Nume', inplace=True)
#
# #apelare in functie de coloana / in functie de index
# print(df.loc['Bogdan'])
# print(df.iloc[1])
#
# # selectarea mai multor coloane dintr-un pandas
#
# print(df.loc[['Ana', 'Cristina'], ['Varsta', 'Salariu']])
#
# df_filtrat = df[df['Varsta'] > 20]
# print(df_filtrat)
#
# df_filtrat_complex = df[(df['Varsta'] >= 25) & (df['Experienta'] >= 2)]
# print(df_filtrat_complex)
#
# # sortarea coloanelor
# df_sortat = df.sort_values('Varsta', ascending=True)
# print(df_sortat)




# # cu coloane duplicate
# data = {'Nume':['Ana', 'Bogdan', 'Ana', 'Cristina', 'David', 'Ana'],
#        'Varsta': [25,30,25,22,35,25],
#        'Salariu': [50000, 60000, 50000, 45000, 70000, 50000]}
# df = pd.DataFrame(data)
# print(df)
# print('=====================')
#
# # cum eliminam datele duplicate
# df_fara_duplicate = df.drop_duplicates()
# print(df_fara_duplicate)

# data = {'Nume': ['Ana', 'Bogdan', None, 'Cristina', 'David'],
#         'Varsta': [25, 38, None, 22, 35],
#         'Salariu': [50000, None, 45000, 70000, 60000]}
#
# df = pd.DataFrame(data)
# print(df)
# print('===============')
# #elimina liniile pe care avem None
# df_fara_none = df.dropna()
# print(df_fara_none)
#
# df_cu_zero = df.fillna(0)
# print(df_cu_zero)
#
# # a = 2
# # b = 3
# # c = 4
# # d = 0
# # print((a+b+c+d)/4)
#
# # in momentul in care avem None recomandabil este sa se elimine aceste date
# # deoarece ne altereaza interpretarea datelor si modelelor
# #Se recomanda inlocuirea valorilor None cu media celorlalte nomere
# # atunci cand avem multe date lipsa intr-o baza de date
#
# df['Experienta'] = [2, 5, 1, 3, 4]
# df_redenumire = df.rename(columns= {'Nume': 'Numele', 'Varsta': 'Varsta', 'Salariu': 'Salariul'})
# print(df_redenumire)
#
# #Redenumeste coloanele
# df.rename(columns={'Nume': 'Numele', 'Varsta': 'Varsta', 'Salariu': 'Salariul'}, inplace=True)
# print(df)

# # Gropeaza in functie de valorile unui caloane (in cazul asta in 2 grupuri "IT" si 'HR')
# data = {'Nume': ['Ana', 'Bogdan', 'Cristina', 'David', 'Elena', 'Florin'],
#         'Varsta': [25, 30, 22, 35, 28, 40],
#         'Salariu': [50000, 60000, 45000, 70000, 55000, 80000],
#         'Departamente': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR']}
# df = pd.DataFrame(data)
# print(df)
# print('++++++++++')
# grupuri_departamente = df.groupby('Departamente')
# for nume_departament, grup in grupuri_departamente:
#         print(grup)
# # face media salariilor in functie de gruparile facute anterior
# medie_salarii = grupuri_departamente['Salariu'].mean()
# print(medie_salarii)
#
# # face modificari/calcule in fuctie de grupurile selectate anterior (IT, HR)
# rezultate_agregare = grupuri_departamente.agg({'Varsta': 'mean', 'Salariu': ['sum', 'median'], 'Nume': 'count'})
# print(rezultate_agregare)

# data = {'Data': ['2022-01-01', '2022-02-01', '2022-03-01'],
#         'Vanzari': [100, 150, 200]}
#
# df = pd.DataFrame(data)
# print(df)
# df['Data']= pd.to_datetime(df['Data'])
# # # Observatie!!!:  NU MERGE - df['Zi'] = df['Data'].dt.day
# # # Trebuie convertit cu datetime
# df['Data'] = pd.to_datetime(df['Data'])
# print(df.dtypes)
# df['Zi'] = df['Data'].dt.day
# print(df)
#
# df['Luna'] = df['Data'].dt.month
# print(df)
# df['An'] = df['Data'].dt.year
# print(df)
#
# df['Diferenta_zi'] = (df['Data'] - pd.to_datetime('2022-01-01')).dt.days
# print(df)


# # concatentarea a doua data frame
# # ignore_index=True - face indexul corect fara el ar avea valorile 0, 1 + 0, 1 pentru ca se concateneaza doua frameuri
#
# df1 = pd.DataFrame({'Nume':['Ana', 'Bogdan'],
#                     'Varsta': [25,30]})
# df2 = pd.DataFrame({'Nume':['Cristina', 'David'],
#                     'Varsta': [22,35]})
# df_concat_randuri = pd.concat([df1, df2], ignore_index=True)
# print(df_concat_randuri)
#
# df_concat_coloane = pd.concat([df1, df2], axis=1)
# print(df_concat_coloane)

# afiseaza persoanele care au acelasi ID
# how='inner' - pastram randurile cu indexurile comune
# how='outer' - afiseaza tot

# df1 = pd.DataFrame({'ID': [1,2, 3],
#                     'Nume':['Ana', 'Bogdan', 'Silviu'],
#                     'Varsta': [25,30,34]}, index = [1,2,3])
# df2 = pd.DataFrame({'ID': [2,3, 4],
#                     'Nume':['Cristina', 'David', 'Maria'],
#                     'Varsta': [22,35,21]}, index = [1,2,3])
#
# df_merge = pd.merge(df1, df2, on='ID', how='inner')
# print(df_merge)
# df_merge = pd.merge(df1, df2, on='ID', how='outer')
# print(df_merge)

# df1 = pd.DataFrame({'Nume':['Ana', 'Bogdan', 'Silviu'],
#                     'Varsta': [25,30,34]},
#                    index = [1,2,3])
# df2 = pd.DataFrame({'Experienta':[2,3,4],
#                     'Departament': ['HR', 'IT', 'Sales']},
#                    index = [2,3,4])
#
# df_join = df1.join(df2, how='inner')
# print(df_join)

data = {'Nume': ['Ana', 'Bogdan', 'Cristina', 'David', 'Elena', 'Florin'],
        'Varsta': [25, 30, 22, 35, 28, 40],
        'Salariu': [50000, 60000, 45000, 70000, 55000, 80000],
        'Departamente': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR']}

df = pd.DataFrame(data)
# print(df)
# df.to_csv('date.csv', index=False)

# df = pd.read_csv('date.csv')
# print(df)

#df.to_csv('date_cu_index.csv')
df = pd.read_csv('date_cu_index.csv')
print(df)















