import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Charger les datasets
df_original = pd.read_excel("dataset_modifie.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_copulagan = pd.read_excel("dataset_synthetique_copulagan2.xlsx")

# Colonne cible
target_col = "Sample_Type"


# Séparer X et y
def prepare_data(df):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


x_original, y_original = prepare_data(df_original)
x_tvae, y_tvae = prepare_data(df_tvae)
x_ctgan, y_ctgan = prepare_data(df_ctgan)
x_copulagan, y_copulagan = prepare_data(df_copulagan)


# Division train/test
def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)


x_train_orig, x_test_orig, y_train_orig, y_test_orig = split_data(x_original, y_original)
x_train_tvae, x_test_tvae, y_train_tvae, y_test_tvae = split_data(x_tvae, y_tvae)
x_train_ctgan, x_test_ctgan, y_train_ctgan, y_test_ctgan = split_data(x_ctgan, y_ctgan)
x_train_copulagan, x_test_copulagan, y_train_copulagan, y_test_copulagan = split_data(x_copulagan, y_copulagan)


# Définition des modèles de base
def get_base_models():
    models = dict()

    # XGBoost
    models['xgb'] = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # MLP avec standardisation
    models['mlp'] = make_pipeline(
        MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=42
        )
    )

    # SVM avec standardisation
    models['svm'] = make_pipeline(
        SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Nécessaire pour StackingClassifier
            random_state=42
        )
    )

    return models


# Définition du modèle méta (final)
def get_meta_model():
    return XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )


# Création du modèle stacké
def get_stacked_model():
    base_models = get_base_models()
    meta_model = get_meta_model()

    stacked_model = StackingClassifier(
        estimators=list(base_models.items()),
        final_estimator=meta_model,
        stack_method='auto',
        n_jobs=-1,
        passthrough=False
    )

    return stacked_model


# Entraînement et évaluation
def train_stacked_model(x_train, x_test, y_train, y_test, dataset_name):
    model = get_stacked_model()

    print(f"\n🔨 Entraînement du modèle stacké sur {dataset_name}...")
    model.fit(x_train, y_train)

    print("📊 Évaluation en cours...")
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]  # Pour AUC-ROC

    acc = accuracy_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n📊 Résultats pour {dataset_name} (modèle stacké) :")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ Recall Score : {rs:.4f}")
    print(f"✅ Precision Score : {ps:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")
    print(f"✅ AUC-ROC  : {auc:.4f}")

    return model


# Évaluations
print("🚀 Début des évaluations avec modèle stacké (XGBoost + MLP + SVM)")
train_stacked_model(x_train_orig, x_test_orig, y_train_orig, y_test_orig, "Dataset Original")
train_stacked_model(x_train_ctgan, x_test_ctgan, y_train_ctgan, y_test_ctgan, "Dataset CTGAN")
train_stacked_model(x_train_tvae, x_test_tvae, y_train_tvae, y_test_tvae, "Dataset TVAE")
train_stacked_model(x_train_copulagan, x_test_copulagan, y_train_copulagan, y_test_copulagan, "Dataset CopulaGAN")