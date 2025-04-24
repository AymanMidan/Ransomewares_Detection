import pandas as pd

# 1. Charger les données originales (NE PAS MODIFIER)
df_original = pd.read_excel('dataset_normalized_robust.xlsx')  # ou .excel
high_corr_pairs = pd.read_csv('high_correlation_pairs.csv')

# 2. Créer une copie pour le dataset réduit
df_reduit = df_original.copy()

# 3. Supprimer AUTOMATIQUEMENT les APIs redondantes (colonne 'API_2')
apis_a_supprimer = high_corr_pairs['API_2'].unique()
df_reduit = df_reduit.drop(columns=apis_a_supprimer)

# 4. Fusion MANUELLE des paires logiques (exemples)
fusion_apis = {
    # Format : {'nouvelle_feature': ['API1', 'API2']}
    'Socket_Ops': ['bind', 'WSASocketW'],
    'Crypto_Ops': ['CryptCreateHash', 'CryptHashData'],
    'Process_Monitor': ['CreateToolhelp32Snapshot', 'Process32FirstW']
}

for nouvelle_col, apis in fusion_apis.items():
    if all(api in df_reduit.columns for api in apis):  # Vérifie si les APIs existent
        df_reduit[nouvelle_col] = df_reduit[apis].sum(axis=1)
        df_reduit = df_reduit.drop(columns=apis)

# 5. Sauvegarde en NOUVEAU fichier (sans écraser l'original)
df_reduit.to_excel('dataset_reduit_et_normaliser.xlsx', index=False)

from sklearn.decomposition import PCA

# Vérification de la variance restante après réduction
pca_original = PCA(n_components=0.95)  # Conserver 95% de la variance
pca_reduit = PCA(n_components=0.95)

pca_original.fit(df_original.drop(columns=['Sample_Type'], errors='ignore'))
pca_reduit.fit(df_reduit.drop(columns=['Sample_Type'], errors='ignore'))

print("\n=== Évaluation de la perte d'information ===")
print(f"Nombre de composantes PCA (original) : {pca_original.n_components_}")
print(f"Nombre de composantes PCA (réduit)   : {pca_reduit.n_components_}")
print(f"Variance expliquée (original) : {sum(pca_original.explained_variance_ratio_):.4f}")
print(f"Variance expliquée (réduit)   : {sum(pca_reduit.explained_variance_ratio_):.4f}")

if sum(pca_reduit.explained_variance_ratio_) < 0.90:
    print("\n⚠️ Attention : Trop d'information a été perdue, réévaluer la réduction.")

# 6. Rapport des modifications
print("=== Réduction terminée ===")
print(f"APIs originales : {len(df_original.columns)}")
print(f"APIs supprimées : {len(apis_a_supprimer)}")
print(f"APIs restantes : {len(df_reduit.columns)}")
print(f"Nouvelles features créées : {list(fusion_apis.keys())}")
print("\nDataset sauvegardé sous 'dataset_reduit.csv'")