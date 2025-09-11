import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


# Prima si separa il test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Poi si separa il validation dal resto
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)



### Modello DecisionTreeClassifier

# Alleno un modello DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)

# Valutazione sul validation set (per tuning)
y_val_pred = tree.predict(X_val)
print("Valutazione sul Validation set per Decision Tree:")
print(classification_report(y_val, y_val_pred, digits=3))

# Valutazione sul test set
y_test_pred = tree.predict(X_test)
print("\nValutazione sul Test set per Decision Tree:")
print(classification_report(y_test, y_test_pred, digits=3))

# Accuracy sul test set
accuracy = accuracy_score(y_test, y_test_pred)
print(f"\nAccuracy (Test) per Decision Tree: {accuracy:.3f}")



### Modello RandomForestClassifier


# Addestramento Random Forest

rf = RandomForestClassifier(
    n_estimators=100,      # numero di alberi
    max_depth=None,        # profondità massima (None = crescita completa)
    random_state=42,
    n_jobs=-1              # usa tutti i core della CPU
)
rf.fit(X_train, y_train)


# Valutazione sul validation set

y_val_pred = rf.predict(X_val)
print("Valutazione sul Validation set per Random Forest:")
print(classification_report(y_val, y_val_pred, digits=3))
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Accuracy (Validation): {val_accuracy:.3f}\n")


# Valutazione sul test set

y_test_pred = rf.predict(X_test)
print("Valutazione sul Test set per Random Forest:")
print(classification_report(y_test, y_test_pred, digits=3))
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy (Test): {test_accuracy:.3f}\n")


