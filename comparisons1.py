import pandas as pd
from scipy.stats import ks_2samp

# 📁 Charger les datasets
df_original = pd.read_excel("dataset_modifie.xlsx")
df_tvae = pd.read_excel("dataset_synthetique_tvae.xlsx")
df_ctgan = pd.read_excel("dataset_synthetique_ctgan.xlsx")
df_copulagan = pd.read_excel("dataset_synthetique_copulagan.xlsx")

# 🎯 Colonnes à comparer (remplacez par vos 3 colonnes d'intérêt)
cols_to_check = ["GetFileVersionInfoW", "ioctlsocket", "GetUserNameA"]

# ✅ Vérification de l'existence des colonnes
datasets = {
    "Original": df_original,
    "CopulaGAN": df_copulagan,
    "TVAE": df_tvae,
    "CTGAN": df_ctgan
}

for col in cols_to_check:
    for name, df in datasets.items():
        if col not in df.columns:
            print(f"❌ Erreur : '{col}' absente dans le dataset {name}.")
            exit()

# 📊 Test KS pour chaque colonne
print("\n🔎 Résultats des tests de Kolmogorov-Smirnov (KS) :")

for col in cols_to_check:
    print(f"\n📌 Colonne : '{col}'")

    for name, df in datasets.items():
        if name != "Original":
            ks_stat, p_value = ks_2samp(df_original[col], df[col])
            print(f"\n🔹 Comparaison Original vs {name}")
            print(f"   KS Statistic : {ks_stat:.4f}")
            print(f"   P-value      : {p_value:.4f}")
            if p_value > 0.05:
                print("   ✅ Distributions similaires (H0 non rejetée)")
            else:
                print("   ❌ Distributions différentes (H0 rejetée)")

# 📊 Vérification de la répartition des classes (optionnel)
print("\n\n📌 Répartition des classes (Sample_Type) :")

for name, df in datasets.items():
    ransomware_count = len(df[df['Sample_Type'] == 1])
    goodware_count = len(df[df['Sample_Type'] == 0])
    total = len(df)
    print(f"\n🔹 {name} :")
    print(f"   Ransomwares : {ransomware_count} ({ransomware_count / total * 100:.2f}%)")
    print(f"   Goodwares   : {goodware_count} ({goodware_count / total * 100:.2f}%)")