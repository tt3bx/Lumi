import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("sign_data.csv")

X = data.drop("label", axis=1)
y = data["label"]

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "lumisign_model.pkl")

print("Model saved as lumisign_model.pkl")
print("Classes:", model.classes_)