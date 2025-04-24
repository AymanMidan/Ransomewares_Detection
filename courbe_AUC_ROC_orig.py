import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_excel("dataset_synthetique_ctgan.xlsx")
X = data.drop(columns=["Sample_Type"])
y = data['Sample_Type']

# Diviser le dataset en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

# Prédire les probabilités pour la classe positive
y_probs = xgb_model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive

# Calculer la courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Visualiser la courbe ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonale pour la courbe aléatoire
plt.title('Courbe AUC-ROC')
plt.xlabel('Taux de Faux Positifs (FPR)')
plt.ylabel('Taux de Vrais Positifs (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Afficher l'AUC
print(f"AUC-ROC: {roc_auc:.4f}")