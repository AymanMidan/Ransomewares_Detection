import pandas as pd
import numpy as np

# 1. Charger le dataset généré
df_copula = pd.read_excel("dataset_synthetique_copulagan3.xlsx")

# 2. Appliquer le bruit uniquement sur les colonnes numériques
for col in df_copula.columns:
    if col != "Sample_Type":  # Exclure la colonne cible
        df_copula[col] = df_copula[col] + np.random.normal(0, 0.01, len(df_copula))

# 3. Sauvegarder la version bruitée
df_copula.to_excel("dataset_synthetique_copulagan_bruite.xlsx", index=False)
print("✅ Bruit ajouté avec succès !")