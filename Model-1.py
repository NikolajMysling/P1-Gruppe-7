import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# === Indlæs data ===
student_data = pd.read_csv("student_lifestyle_dataset.csv")

print(student_data.info())
print(student_data.head())

# === Mål og features ===
TARGET = "GPA"                  # Vi forudsiger GPA (kontinuerlig)
DROP_COLS = ["Student_ID"]      # ID skal ikke bruges som feature

X = student_data.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = student_data[TARGET]

# === Kolonnelister ===
num_feats = X.select_dtypes(include=["number"]).columns.tolist()
cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

# === Preprocessing ===
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_feats),
    ("cat", cat_pipe, cat_feats)
])

# === Model ===
model = Pipeline([
    ("preprocess", preprocess),
    ("reg", LinearRegression())
])

# === Split + CV ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
print("\nCross-validation (Linear Regression on GPA):")
print("Scores:", scores)
print("Mean R^2:", scores.mean())
print("Std:", scores.std())

# Beregn korrelation (kun numeriske kolonner)
corr_matrix = student_data.select_dtypes(include=[np.number]).corr()

# Udskriv korrelation med GPA
print(corr_matrix["GPA"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix, cmap="coolwarm", fignum=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap (Student Lifestyle Data)", pad=20)
plt.show()