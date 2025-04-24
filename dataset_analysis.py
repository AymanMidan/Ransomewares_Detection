import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import mannwhitneyu

# Chargement du dataset
df = pd.read_excel('dataset.xlsx')

# Conversion de Sample_Type en valeurs numériques
df["Sample_Type"] = df["Sample_Type"].map({"G": 0, "R": 1})

# Vérification initiale
print("Informations de base du dataset:")
print(df.info())

# Vérification de l'équilibre Ransomware vs Goodware
type_counts = df['Sample_Type'].value_counts()
print("\nRépartition des types:")
print(type_counts)

# Visualisation de la répartition
plt.figure(figsize=(8, 6))
type_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Répartition Ransomware vs Goodware')
plt.xlabel('Type')
plt.ylabel("Nombre d'instances")
plt.xticks(ticks=[0, 1], labels=['Goodware', 'Ransomware'], rotation=0)
plt.tight_layout()
plt.savefig('type_distribution.png')
plt.close()

# Sélectionner 50 colonnes aléatoires (en excluant 'Sample_Type')
all_columns = df.columns.drop('Sample_Type')
random_columns = random.sample(list(all_columns), min(50, len(all_columns)))

# Statistiques descriptives générales sur les 50 colonnes sélectionnées
print("\nStatistiques descriptives générales (50 colonnes aléatoires):")
print(df[random_columns].describe())

# Statistiques séparées pour Goodware et Ransomware sur les 50 colonnes sélectionnées
print("\nStatistiques descriptives pour les Ransomwares (50 colonnes aléatoires):")
print(df.loc[df['Sample_Type'] == 1, random_columns].describe())

print("\nStatistiques descriptives pour les Goodwares (50 colonnes aléatoires):")
print(df.loc[df['Sample_Type'] == 0, random_columns].describe())

# Vérification des valeurs manquantes (toutes colonnes)
print("\nValeurs manquantes (toutes colonnes):")
print(df.isnull().sum())

# Analyse de la distribution des APIs (toutes colonnes)
api_usage = df.drop('Sample_Type', axis=1)
api_usage_mean = api_usage.mean().sort_values(ascending=False)

# Visualisation des APIs les plus utilisées (top 30)
plt.figure(figsize=(18, 8))
bars = api_usage_mean[:30].plot(kind='bar', color='purple')
plt.title('Top 30 des APIs les plus utilisées', pad=20)
plt.xlabel('APIs')
plt.ylabel('Moyenne d\'utilisation')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('api_usage_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Boxplot des 5 APIs les plus utilisées par classe
top_5_apis = api_usage_mean.head(5).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.melt(id_vars='Sample_Type', value_vars=top_5_apis),
            x='variable', y='value', hue='Sample_Type')
plt.yscale('log')
plt.title('Distribution des Top 5 APIs par Classe')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_apis_boxplot.png', dpi=300)
plt.close()

# Test de Mann-Whitney pour les top APIs
print("\nTests de Mann-Whitney pour les top APIs:")
for api in top_5_apis:
    stat, p = mannwhitneyu(df.loc[df['Sample_Type'] == 1, api],
                           df.loc[df['Sample_Type'] == 0, api])
    print(f"{api}: p-value = {p:.4f} → {'Différence significative' if p < 0.05 else 'Pas de différence'}")

# Matrice de corrélation et paires fortement corrélées
corr_matrix = api_usage.corr()

# Heatmap des corrélations (top 5 APIs)
plt.figure(figsize=(15, 12))
sns.heatmap(api_usage[top_5_apis].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Corrélations entre Top 5 APIs')
plt.tight_layout()
plt.savefig('top_apis_heatmap.png', dpi=300)
plt.close()


# Extraction des paires fortement corrélées
threshold = 0.95
high_corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ["API_1", "API_2", "Correlation"]
high_corr_pairs = high_corr_pairs.loc[
    (high_corr_pairs["Correlation"] >= threshold) |
    (high_corr_pairs["Correlation"] <= -threshold)
]

print("\nPaires d'APIs avec une forte corrélation :")
print(high_corr_pairs.sort_values(by="Correlation", ascending=False))
high_corr_pairs.to_csv("high_correlation_pairs.csv", index=False)

# Séparation des features et labels
x = df.drop('Sample_Type', axis=1)
y = df['Sample_Type']
print("\nFormes des données:")
print("x shape:", x.shape)
print("y shape:", y.shape)
