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


"""
OUTPUT ATTESO:

Valutazione sul Validation set per Decision Tree:
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000         8
Iris-versicolor      0.889     1.000     0.941         8
 Iris-virginica      1.000     0.857     0.923         7

       accuracy                          0.957        23
      macro avg      0.963     0.952     0.955        23
   weighted avg      0.961     0.957     0.956        23


Valutazione sul Test set per Decision Tree:
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000         7
Iris-versicolor      0.778     0.875     0.824         8
 Iris-virginica      0.857     0.750     0.800         8

       accuracy                          0.870        23
      macro avg      0.878     0.875     0.875        23
   weighted avg      0.873     0.870     0.869        23


Accuracy (Test) per Decision Tree: 0.870
Valutazione sul Validation set per Random Forest:
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000         8
Iris-versicolor      0.889     1.000     0.941         8
 Iris-virginica      1.000     0.857     0.923         7

       accuracy                          0.957        23
      macro avg      0.963     0.952     0.955        23
   weighted avg      0.961     0.957     0.956        23

Accuracy (Validation): 0.957

Valutazione sul Test set per Random Forest:
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000         7
Iris-versicolor      0.875     0.875     0.875         8
 Iris-virginica      0.875     0.875     0.875         8

       accuracy                          0.913        23
      macro avg      0.917     0.917     0.917        23
   weighted avg      0.913     0.913     0.913        23

Accuracy (Test): 0.913

"""
