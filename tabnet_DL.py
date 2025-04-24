import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import torch

# ✅ Charger les datasets
df_original = pd.read_excel("dataset_modifie.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_copulaGAN = pd.read_excel("dataset_synthetique_copulagan2.xlsx")

# ✅ Définir la colonne cible
target_col = "Sample_Type"

# ✅ Préparation des données
def prepare_data(df):
    x = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return x, y

x_original, y_original = prepare_data(df_original)
x_tvae, y_tvae = prepare_data(df_tvae)
x_ctgan, y_ctgan = prepare_data(df_ctgan)
x_copulaGAN, y_copulaGAN = prepare_data(df_copulaGAN)

# ✅ Division train/test
def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

x_train_orig, x_test_orig, y_train_orig, y_test_orig = split_data(x_original, y_original)
x_train_tvae, x_test_tvae, y_train_tvae, y_test_tvae = split_data(x_tvae, y_tvae)
x_train_ctgan, x_test_ctgan, y_train_ctgan, y_test_ctgan = split_data(x_ctgan, y_ctgan)
x_train_copulaGAN, x_test_copulaGAN, y_train_copulaGAN, y_test_copulaGAN = split_data(x_copulaGAN, y_copulaGAN)

# ✅ Fonction d'entraînement et évaluation
def train_tabnet(x_train, x_test, y_train, y_test, dataset_name):
    model = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5,
        gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
        seed=42
    )

    # Convertir les données en float32
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    # Entraînement
    model.fit(
        X_train=x_train, y_train=y_train,
        eval_set=[(x_test, y_test)],
        eval_metric=['accuracy'],
        max_epochs=200,
        patience=20,
        batch_size=256,
        virtual_batch_size=128,
        num_workers=0
    )

    # Prédictions
    y_pred = model.predict(x_test)

    # Évaluation
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n📊 Résultats pour {dataset_name} (TabNet) :")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Recall Score : {rec:.4f}")
    print(f"✅ Precision Score : {prec:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")
    print(f"✅ AUC-ROC  : {auc:.4f}")

# ✅ Entraîner et tester TabNet sur les deux datasets
train_tabnet(x_train_orig, x_test_orig, y_train_orig, y_test_orig, "Dataset Original")
train_tabnet(x_train_ctgan, x_test_ctgan, y_train_ctgan, y_test_ctgan, "Dataset CTGAN")
train_tabnet(x_train_tvae, x_test_tvae, y_train_tvae, y_test_tvae, "Dataset TVAE")
train_tabnet(x_train_copulaGAN, x_test_copulaGAN, y_train_copulaGAN, y_test_copulaGAN, "Dataset CopulaGAN")

