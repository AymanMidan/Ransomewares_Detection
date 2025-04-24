from scipy.stats import shapiro
import pandas as pd
import numpy as np

# 1. Charger les données
df = pd.read_excel('dataset.xlsx')
x = df.drop('Sample_Type', axis=1)

# 2. Sélectionner 100 colonnes aléatoires
np.random.seed(42)
cols_to_test = np.random.choice(x.columns, size=100, replace=False)

# 3. Tester la normalité
normal_count = 0

for col in cols_to_test:
    # Prendre un échantillon (car Shapiro limite à 5000 points)
    sample = x[col].sample(n=min(5000, len(x)), random_state=42) if len(x) > 5000 else x[col]

    # Test Shapiro-Wilk
    _, p = shapiro(sample)

    # Afficher les résultats
    print(f"{col[:25]:<25} | p-value = {p:.4f} | {'Normal' if p > 0.05 else 'Non-normal'}")

    if p > 0.05:
        normal_count += 1

# 4. Recommandation finale
print("\n=== Résumé ===")
print(f"Colonnes normales    : {normal_count}/100")
print(f"Colonnes non-normales: {100 - normal_count}/100")

if normal_count >= 70:
    print("\nRecommandation : StandardScaler (majorité normale)")
elif normal_count >= 30:
    print("\nRecommandation : PowerTransformer (mixte normal/non-normal)")
else:
    print("\nRecommandation : RobustScaler (majorité non-normale)")
