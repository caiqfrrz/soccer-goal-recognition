"""Step 4: simplest baseline - logistic regression on time-averaged windows."""
from sklearn.metrics import classification_report

from goalspot.data import load_windows
from goalspot.models import make_logreg

X_train, y_train, _ = load_windows("train")
X_valid, y_valid, _ = load_windows("valid")
print("train:", X_train.shape, "| valid:", X_valid.shape)

model = make_logreg()
model.fit(X_train.mean(axis=1), y_train)              # (N, 40, D) -> (N, D)
pred = model.predict(X_valid.mean(axis=1))
print(classification_report(y_valid, pred, target_names=["no goal", "goal"], digits=3))
