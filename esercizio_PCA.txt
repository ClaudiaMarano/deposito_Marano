import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

### Caricamento del dataset Digit Recognizer

# Percorso file
file_path = r"C:\Users\BZ464JU\OneDrive - EY\Documents\Academy\train.csv"

# Caricamento del dataset
df = pd.read_csv(file_path)

# Prime righe
print(df.head())



### Preprocessing

# Standardizzazione

scaler = StandardScaler()
X = scaler.fit_transform(df.drop(columns=["label"]))



### Riduzione dimensionale con PCA

# Calcolo quante componenti servono per il 95% della varianza
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
print(f"Numero di componenti per il 95% della varianza: {pca.n_components_}")

# Trasformazione PCA utilizzando il numero di componenti calcolato
X_reduced = PCA(n_components=pca.n_components_).fit_transform(X)
print(f"Forma dei dati dopo PCA: {X_reduced.shape}")
# Prime 5 righe dei dati ridotti
print(X_reduced[:5])

### Split train-test

# Split dei dati in train e test (80% train, 20% test, con stratify = y)

y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=42)



### Training di un DecisionTreeClassifier sui dati ridotti

# Alleno un DecisionTreeClassifier sui dati ridotti

tree = DecisionTreeClassifier(max_depth=10, random_state=42)
tree.fit(X_train, y_train)
y_test_pred = tree.predict(X_test)

# Valutazione sul test set: accuracy e matrice di confusione

accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nAccuratezza per Decision Tree con PCA: {accuracy:.3f}")
print("Matrice di confusione:")
print(confusion_matrix(y_test, y_test_pred))



### Training di un DecisionTreeClassifier sui dati originali (non ridotti)

# Alleno un DecisionTreeClassifier sui dati originali (non ridotti)
tree_orig = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_orig.fit(X_train, y_train)
y_test_pred_orig = tree_orig.predict(X_test)

# Valutazione sul test set: accuracy e matrice di confusione

accuracy_orig = accuracy_score(y_test, y_test_pred_orig)
print(f"\nAccuratezza per Decision Tree con dati originali: {accuracy_orig:.3f}")
print("Matrice di confusione:")
print(confusion_matrix(y_test, y_test_pred_orig))


"""
OUTPUT ATTESO:
Numero di componenti per il 95% della varianza: 320

Accuratezza per Decision Tree con PCA: 0.783
Matrice di confusione:
[[692   0  16   7  13  48  21   6   8  16]
 [  0 857  20  21   5  13   6   4  10   1]
 [  9   2 661  16  25   9  26  13  59  15]
 [  6   8  67 620  12  58  14   8  70   7]
 [  7   4  11  23 604  12  17  18  23  95]
 [ 14   0  23  67  31 523  17  17  44  23]
 [ 10   0  15   5  21  15 748   0  10   3]
 [  6   4  16   9  26  19   0 685  23  92]
 [  8   5  39  71  28  68   8  16 542  28]
 [  4   4   9  28  66  14   5  52  12 644]]

Accuratezza per Decision Tree con dati originali: 0.783
Matrice di confusione:
[[692   0  16   7  13  48  21   6   8  16]
 [  0 857  20  21   5  13   6   4  10   1]
 [  9   2 661  16  25   9  26  13  59  15]
 [  6   8  67 620  12  58  14   8  70   7]
 [  7   4  11  23 604  12  17  18  23  95]
 [ 14   0  23  67  31 523  17  17  44  23]
 [ 10   0  15   5  21  15 748   0  10   3]
 [  6   4  16   9  26  19   0 685  23  92]
 [  8   5  39  71  28  68   8  16 542  28]
 [  4   4   9  28  66  14   5  52  12 644]]

 Da questa analisi risulta evidente che l'uso della PCA non ha migliorato l'accuratezza del modello di Decision Tree. In questo caso specifico
 l'utilizzo della PCA non ha portato un miglioramento significativo delle prestazioni del modello.

"""