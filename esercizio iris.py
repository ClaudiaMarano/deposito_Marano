import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


### Caricamento del dataset Iris

# Percorso del file
file_path = r"C:\Users\BZ464JU\Downloads\archive\iris.data.csv"
# Caricamento del dataset
df = pd.read_csv(file_path, header=None) 


### Preprocessing

# Assegno nomi alle colonne
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Separo feature (X) e target (y)
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["class"]

# Divido il dataset in train e test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


### Modello DecisionTreeClassifier

# Alleno un modello DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Faccio predizioni sul set di test
y_pred = tree.predict(X_test)


### Valutazione del modello

# Stampo l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Stampo il classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))


# Stampo le prime 5 righe
print(df.head())





