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

forest_weight = RandomForestClassifier(class_weight="balanced", random_state=42, n_estimators=50, max_depth=5)

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

forest_smote = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=5)

forest_smote.fit(X_train_smote, y_train_smote)

print("\nRandom Forest con SMOTE")
print(confusion_matrix(y_test, forest_smote.predict(X_test)))
print(classification_report(y_test, forest_smote.predict(X_test)))



"""
OUTPUT ATTESO:

Class
0    0.998273
1    0.001727
Name: proportion, dtype: float64

Decision Tree con class_weight
[[56830    34]
 [   27    71]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.68      0.72      0.70        98

    accuracy                           1.00     56962
   macro avg       0.84      0.86      0.85     56962
weighted avg       1.00      1.00      1.00     56962


Random Forest con class_weight
[[56652   212]
 [   11    87]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.29      0.89      0.44        98

    accuracy                           1.00     56962
   macro avg       0.65      0.94      0.72     56962
weighted avg       1.00      1.00      1.00     56962


Decision Tree con SMOTE
[[56758   106]
 [   20    78]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.42      0.80      0.55        98

    accuracy                           1.00     56962
   macro avg       0.71      0.90      0.78     56962
weighted avg       1.00      1.00      1.00     56962


Random Forest con SMOTE
[[56448   416]
 [   11    87]]
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     56864
           1       0.17      0.89      0.29        98

    accuracy                           0.99     56962
   macro avg       0.59      0.94      0.64     56962
weighted avg       1.00      0.99      1.00     56962

"""


