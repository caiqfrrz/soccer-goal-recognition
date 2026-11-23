import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load(split):
    X = np.load(f"dataset/X_{split}.npy")   # (N, 40, 512)
    y = np.load(f"dataset/y_{split}.npy")   # (N,)  1 = gol, 0 = não gol
    return X, y


# --- Parte 1: carregar as janelas criadas no passo 3 ---
X_train, y_train = load("train")
X_valid, y_valid = load("valid")
print("train:", X_train.shape, "| valid:", X_valid.shape)

# --- Parte 2: resumir cada janela num único vetor (média no tempo) ---
# (N, 40, 512) -> (N, 512)
X_train_flat = X_train.mean(axis=1)
X_valid_flat = X_valid.mean(axis=1)

# --- Parte 3: treinar o modelo ---
model = make_pipeline(
    StandardScaler(),                                    # deixa as 512 colunas na mesma escala
    LogisticRegression(max_iter=1000, class_weight="balanced"),
)
model.fit(X_train_flat, y_train)

# --- Parte 4: avaliar em dados que o modelo nunca viu ---
pred = model.predict(X_valid_flat)
print(classification_report(y_valid, pred, target_names=["não gol", "gol"], digits=3))