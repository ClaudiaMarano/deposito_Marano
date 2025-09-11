import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


### Caricamento del dataset Credit Card Fraud Detection

df = pd.read_csv(r"C:\Users\BZ464JU\Downloads\archive (1)\creditcard.csv")


### Preprocessing

# Separo feature (X) e target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Print della distribuzione delle classi
print(y.value_counts(normalize=True))  # Distribuzione delle classi


### Divido il dataset in train e test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


### UTILIZZO DI CLASS_WEIGHT PER IL BILANCIAMENTO DELLE CLASSI

### Modello DecisionTreeClassifier con bilanciamento delle classi class_weight

tree_weight = DecisionTreeClassifier(class_weight="balanced", random_state=42)

tree_weight.fit(X_train, y_train)

print("\nDecision Tree con class_weight")
print(confusion_matrix(y_test, tree_weight.predict(X_test)))
print(classification_report(y_test, tree_weight.predict(X_test)))



### Modello RandomForestClassifier con bilanciamento delle classi class_weight

forest_weight = RandomForestClassifier(class_weight="balanced", random_state=42)

forest_weight.fit(X_train, y_train)

print("\nRandom Forest con class_weight")
print(confusion_matrix(y_test, forest_weight.predict(X_test)))
print(classification_report(y_test, forest_weight.predict(X_test)))



### UTILIZZO DI SMOTE PER IL BILANCIAMENTO DELLE CLASSI


### Modello Decision Tree con bilanciamento delle classi tramite SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

tree_smote = DecisionTreeClassifier(random_state=42)

tree_smote.fit(X_train_smote, y_train_smote)

print("\nDecision Tree con SMOTE")
print(confusion_matrix(y_test, tree_smote.predict(X_test)))
print(classification_report(y_test, tree_smote.predict(X_test)))


### Modello Random Forest con bilanciamento delle classi tramite SMOTE

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

forest_smote = RandomForestClassifier(random_state=42)

forest_smote.fit(X_train_smote, y_train_smote)

print("\nRandom Forest con SMOTE")
print(confusion_matrix(y_test, forest_smote.predict(X_test)))
print(classification_report(y_test, forest_smote.predict(X_test)))



