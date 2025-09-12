import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns



### Caricamento del dataset Mall Customers

# Percorso del file
file_path = "C:/Users/BZ464JU/Downloads/archive (2)/Mall_Customers.csv"

# Caricamento del dataset
df = pd.read_csv(file_path)

### Preprocessing

# Mantengo solo le colonne Annual Income e Spending Score
df = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Rinomino per comodità
df.columns = ["Annual Income", "Spending Score"]

# Standardizzazione
scaler = StandardScaler()
X = scaler.fit_transform(df)

### Calcolo l'indice silhouette per diversi valori di k

# Calcolo la Silohuette per diversi valori di k

silhouette_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

# Grafico silhouette score per diversi k
plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
plt.title("Silhouette Score al variare di k")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()


# Da questo grafico scelgo k=5

# Applichiamo k-Means ai dati generati
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Visualizzazione con i cluster trovati
plt.figure(figsize=(6, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroidi')
plt.title("Cluster trovati con k-Means")
plt.xlabel("Annual Income (standardizzato)")
plt.ylabel("Spending Score (standardizzato)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


"""
Il Cluster in giallo indica la presenza di clienti che spendono molto, nonostante il reddito basso, e che quindi indica un atteggiamento inatteso. 
Il Cluster è composto da clienti con un reddito basso, ma che effettua meno acquisti rispetto al cluster precedente.
Il Cluster viola indica un gruppo di persone con reddito limitato e che, in quanto tale, che effettua spese limitate. 
Il cluster verde è composto da un gruppo di persone con alto reddito e che effettua molti acquisti. 
"""



