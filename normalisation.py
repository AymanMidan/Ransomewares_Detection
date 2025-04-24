from sklearn.preprocessing import RobustScaler
import pandas as pd

# 1. Charger les données
df = pd.read_excel('dataset.xlsx')

# 2. Séparation features/target
X = df.drop('Sample_Type', axis=1)
y = df['Sample_Type']

# 3. Initialisation du RobustScaler
scaler = RobustScaler()

# 4. Application et conversion en DataFrame
X_normalized = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# 5. Vérification CORRECTE des résultats
print("=== Avant normalisation ===")
print("Médiane et IQR typiques :")
print(X.median().head(15))
print("\nIQR (Q3-Q1) :")
print((X.quantile(0.75) - X.quantile(0.25)).head(15))

print("\n=== Après RobustScaler ===")
print("Médiane normalisée :")
print(X_normalized_df.median().head(15))  # Doit être proche de 0
print("\nIQR normalisé :")
print((X_normalized_df.quantile(0.75) - X_normalized_df.quantile(0.25)).head(15))  # Doit être proche de 1

# Vérification de la variance après normalisation
variances_original = X.var()
variances_normalized = X_normalized_df.var()

# Comparaison avant/après
variance_comparison = pd.DataFrame({
    'Variance Originale': variances_original,
    'Variance Normalisée': variances_normalized
})
variance_comparison['Ratio'] = variance_comparison['Variance Normalisée'] / variance_comparison['Variance Originale']

print("\n=== Vérification de la diversité après normalisation ===")
print(variance_comparison.sort_values(by='Ratio', ascending=True).head(10))  # Afficher les features les plus affectées

# Vérifier si certaines features deviennent trop homogènes
low_variance_features = variance_comparison[variance_comparison['Variance Normalisée'] < 0.01]
print(f"\n⚠️ {len(low_variance_features)} features ont une très faible variance après normalisation !")
if not low_variance_features.empty:
    print("Features concernées :", list(low_variance_features.index))

# Dans normalisation.py, avant la sauvegarde :
low_var_cols = variance_comparison[variance_comparison['Variance Normalisée'] < 0.01].index
X_normalized_df = X_normalized_df.drop(columns=low_var_cols)
print(f"\nSuppression de {len(low_var_cols)} features à faible variance.")

# Réinsérer Sample_Type en première colonne
y_numeric = y.map({'R': 1, 'G': 0})  # Conversion des labels
X_normalized_df.insert(0, 'Sample_Type', y_numeric)

# Sauvegarde
try:
    X_normalized_df.to_excel('dataset_normalized_robust.xlsx', index=False)
    print("\n✅ Sauvegarde réussie dans 'dataset_normalized_robust.csv'")
except PermissionError:
    print("\n❌ ERREUR : Fermez le fichier CSV avant de sauvegarder")
except Exception as e:
    print(f"\n❌ Erreur inattendue : {str(e)}")


# Tout marche bien sauf l'API 'HKEY_CURRENT_USER\\Software\\Microsoft\\CommandProcessor'
# de variance très faible, ce qui est vrai d'apres le dataset original, toutes ces valeurs sont à 0
# Donc cette feature n'importe aucune information (elle ne change pas entre R et G), on doit la supprimer