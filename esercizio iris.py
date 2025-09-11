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



"""
Output atteso:

Accuracy: 0.967

Classification Report:
                 precision    recall  f1-score   support

    Iris-setosa      1.000     1.000     1.000        10
Iris-versicolor      1.000     0.900     0.947        10
 Iris-virginica      0.909     1.000     0.952        10

       accuracy                          0.967        30
      macro avg      0.970     0.967     0.967        30
   weighted avg      0.970     0.967     0.967        30


  sepal_length  sepal_width  petal_length  petal_width        class
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa

"""




