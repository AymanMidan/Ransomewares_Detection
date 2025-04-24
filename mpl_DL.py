import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

# 📂 Chargement des datasets
df_original = pd.read_excel("ctgan_harchali.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_copulaGAN = pd.read_excel("dataset_synthetique_copulagan2.xlsx")

# 🎯 Colonne cible
target_col = "Sample_Type"

# 🔧 Préparation des données
def prepare_data(df):
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y

def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=42)

# 🧠 Fonction d'entraînement et d'évaluation
def train_mlp(x_train, x_test, y_train, y_test, dataset_name):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='tanh',
        solver='adam',
        max_iter=1500,
        random_state=42
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    ps = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    print(f"\n📊 Résultats MLP pour {dataset_name} :")
    print(f"✅ Accuracy        : {acc:.4f}")
    print(f"✅ Recall Score    : {rs:.4f}")
    print(f"✅ Precision Score : {ps:.4f}")
    print(f"✅ F1 Score        : {f1:.4f}")
    print(f"✅ AUC-ROC         : {auc:.4f}")

# 🚀 Entraînement sur les 3 datasets
for df, name in zip([df_original, df_ctgan, df_tvae, df_copulaGAN], ["Dataset Original", "Dataset CTGAN", "Dataset TVAE", "Dataset CopulaGAN"]):
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)
    train_mlp(x_train, x_test, y_train, y_test, name)
