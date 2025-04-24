import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# 💡 Fonction pour entraîner et évaluer un modèle Random Forest
def evaluate_model(df, dataset_name):
    target_col = "Sample_Type"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
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

    print(f"\n📊 Résultats pour {dataset_name} :")
    print(f"✅ Accuracy        : {acc:.4f}")
    print(f"✅ Recall Score    : {recall:.4f}")
    print(f"✅ Precision Score : {precision:.4f}")
    print(f"✅ F1 Score        : {f1:.4f}")
    print(f"✅ AUC-ROC         : {auc:.4f}")

# 📂 Charger les datasets
df_original = pd.read_excel("dataset_modifie.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_CG2 = pd.read_excel("dataset_synthetique_copulagan2.xlsx")


# 📊 Évaluation des modèles sur chaque dataset
evaluate_model(df_original, "Dataset Original")
evaluate_model(df_ctgan, "Dataset CTGAN")
evaluate_model(df_tvae, "Dataset TVAE")
evaluate_model(df_CG2, "Dataset copulaGAN 2")


