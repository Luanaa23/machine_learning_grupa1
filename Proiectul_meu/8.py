# 8. Utilizați setul de date iris și aplicați algoritmul K-Means pentru a grupa
# datele în 3 clustere, bazându-vă pe caracteristicile
# 'sepal length' și 'sepal width'.

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# setului de date Iris cu Seaborn
iris = sns.load_dataset('iris')
print(iris)

# selectez caracteristicile
# 'sepal_length' și 'sepal_width'
X = iris[['sepal_length', 'sepal_width']].values
print(X)

# algoritmul K-Means cu 3 clustere
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X)

# etichetele clusterelor
labels = kmeans.predict(X)

# coordonatelor centroizilor
centers = kmeans.cluster_centers_

# vizualizarea graficului
for i in range(3):
    # selectarea si afisarea punctelor pentru fiecare cluster
    puncte_cluster = X[labels == i]
    plt.scatter(puncte_cluster[:, 0], puncte_cluster[:, 1], label=f'Cluster {i + 1}')
    # puncte_cluster[:, 0], puncte_cluster[:, 1] - coordonatele punctelor
    # ox 'sepal_length' si oy 'sepal_width'

# marchez centroizii pe grafic
plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='X', s=100, label='Centroizi')

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Algoritmul K-Means pe setul de date Iris (3 clustere)')
plt.legend()
plt.show()







