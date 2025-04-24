import pandas as pd

# Charger le fichier Excel dans un DataFrame pandas
try:
    df = pd.read_excel("dataset.xlsx")
except FileNotFoundError:
    print("Erreur : Le fichier 'dataset.xlsx' n'a pas été trouvé.")
    exit()

# Vérifier si la colonne "Sample_Type" existe
if "Sample_Type" not in df.columns:
    print("Erreur : La colonne 'Sample_Type' n'existe pas dans le fichier.")
    exit()

# Remplacer les valeurs dans la colonne "Sample_Type"
df['Sample_Type'] = df['Sample_Type'].replace({'G': 0, 'R': 1})

# Afficher les premières lignes du DataFrame modifié pour vérifier
print("DataFrame après remplacement :")
print(df.head())

# Enregistrer le DataFrame modifié dans un nouveau fichier Excel (ou écraser l'original)
# Pour enregistrer dans un nouveau fichier :
df.to_excel("dataset_modifie.xlsx", index=False)
print("\nLe fichier 'dataset_modifie.xlsx' a été créé avec les remplacements.")

# Pour écraser le fichier original ( soyez prudent avec cette option ! ) :
# df.to_excel("dataset.xlsx", index=False)
# print("\nLe fichier 'dataset.xlsx' a été mis à jour avec les remplacements.")