import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# --- 1. Charger le dataset CTGAN-100k ---
df_ctgan_100k = pd.read_excel("ctgan_harchali.xlsx")
target_col = "Sample_Type"

# --- 2. Séparer les features et la cible ---
X = df_ctgan_100k.drop(columns=[target_col])
y = df_ctgan_100k[target_col]

# --- 3. Division train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Entraînement du modèle XGBoost ---
model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss",
    n_estimators=200, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train)

# --- 5. Évaluation ---
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rs = recall_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print("\n📊 Résultats pour CTGAN-100000 avec XGBoost :")
print(f"✅ Accuracy : {acc:.4f}")
print(f"✅ Recall    : {rs:.4f}")
print(f"✅ Precision : {ps:.4f}")
print(f"✅ F1 Score  : {f1:.4f}")
print(f"✅ AUC-ROC   : {auc:.4f}")

# --- 6. Analyse SHAP ---
print("\n📌 Analyse SHAP en cours...")

# Initialiser l'explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# --- 6.1 Visualisation globale (importance des APIs) ---
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary_ctgan100k.png", dpi=300)
plt.close()
print("📸 SHAP summary bar chart sauvegardé sous 'shap_summary_ctgan100k.png'")

# --- 6.2 Visualisation locale pour un exemple (prédiction détaillée) ---
shap.plots.waterfall(shap_values[0], show=False)
plt.savefig("shap_waterfall_ctgan100k_sample0.png", dpi=300, bbox_inches='tight')
plt.close()
print("📸 SHAP waterfall (exemple) sauvegardé sous 'shap_waterfall_ctgan100k_sample0.png'")
