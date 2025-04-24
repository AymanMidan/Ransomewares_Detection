import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 💡 Fonction pour entraîner et évaluer un modèle LightGBM
def evaluate_lgbm(df, dataset_name):
    target_col = "Sample_Type"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n📊 Résultats LightGBM pour {dataset_name} :")
    print(f"✅ Accuracy        : {acc:.4f}")
    print(f"✅ Recall Score    : {recall:.4f}")
    print(f"✅ Precision Score : {precision:.4f}")
    print(f"✅ F1 Score        : {f1:.4f}")
    print(f"✅ AUC-ROC         : {auc:.4f}")

# 📂 Charger les datasets
df_original = pd.read_excel("dataset_modifie.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_copulaGAN = pd.read_excel("dataset_synthetique_copulagan2.xlsx")

# 📊 Évaluation des modèles LightGBM sur chaque dataset
evaluate_lgbm(df_original, "Dataset Original")
evaluate_lgbm(df_ctgan, "Dataset CTGAN")
evaluate_lgbm(df_tvae, "Dataset TVAE")
evaluate_lgbm(df_copulaGAN, "Dataset copulaGAN")
