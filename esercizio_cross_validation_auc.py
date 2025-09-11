from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

### K-FOLD CROSS-VALIDATION

# K-Fold stratificato
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
cross_tree = cross_val_score(tree, X, y, cv=skf)


print(f"Decision Tree AUC: {cross_tree.mean():.3f} ± {cross_tree.std():.3f}")


"""
OUTPUT ATTESO:
Decision Tree AUC: 0.947 ± 0.027

"""
