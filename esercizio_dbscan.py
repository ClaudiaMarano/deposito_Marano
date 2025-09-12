import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

### Caricamento del dataset

# Percorso del file
file_path = "C:/Users/BZ464JU/Downloads/archive (3)/Wholesale customers data.csv"

# Caricamento del dataset con intestazioni già incluse
df = pd.read_csv(file_path)  

# Tengo solo le feature di spesa (Escludo Channel e Region)
df = df.drop(columns=["Channel", "Region"])
print(df.head())

# Standardizzazione
scaler = StandardScaler()
X = scaler.fit_transform(df)



### Scelta dei parametri per DBSCAN

# Calcolo il k-distance plot per diversi valori di min_samples

for min_samples in [3, 5, 8]:
    neigh = NearestNeighbors(n_neighbors=min_samples)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, -1])
    plt.plot(k_distances, label=f'min_samples={min_samples}')

plt.xlabel('Points sorted by distance')
plt.ylabel('k-distance')
plt.title('k-distance Plot for DBSCAN')
plt.legend()

### Addestramento DBSCAN

# Scelgo eps=1.5 e min_samples=5 in base al grafico
dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(X)

### Valutazione del clustering

# Calcolo silhouette score solo se ci sono almeno 2 cluster e non tutti sono outlier
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
if n_clusters > 1 and n_clusters < len(X) - 1:
    sil_score = silhouette_score(X, labels)
    print(f"Silhouette Score: {sil_score:.3f}")
else:
    print("Silhouette Score not computed: number of clusters not in valid range.")

# Conteggio dei cluster e degli outlier
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
num_outliers = np.sum(labels == -1)
print(f"Numero di cluster trovati: {num_clusters}")
print(f"Numero di outlier (rumore): {num_outliers}")


###  Visualizzazione dei cluster trovati

# Aggiungo le etichette al DataFrame originale
df['Cluster'] = labels
print(df['Cluster'].value_counts())

# Visualizzazione con i cluster trovati
plt.figure(figsize=(8, 6))      
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1 (standardized)')
plt.ylabel('Feature 2 (standardized)')
plt.colorbar(label='Cluster Label')
plt.show()


"""
Interpretazione dei risultati del clustering con Dbscan:
L'algoritmo ha identidicato 2 cluster nei dati. Tramite una rappresentazione 2D i cluster risultano sovrapposti. """