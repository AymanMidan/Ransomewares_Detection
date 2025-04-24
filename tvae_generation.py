import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
import torch
print("🚀 GPU disponible :", torch.cuda.is_available())
print("🔍 Appareil utilisé par PyTorch :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Charger le dataset
file_path = "dataset_reduit_et_normaliser.excel"
df_features = pd.read_excel(file_path)

# Vérifier le dataset
print("Aperçu du dataset original :")
print(df_features.head())

# Créer et détecter les métadonnées
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_features)

metadata.update_column("Sample_Type", sdtype="categorical")

# Forcer certaines colonnes à être numériques si SDV s'est trompé, sauf "Sample_Type"
numerical_columns = df_features.select_dtypes(include=['number']).columns.tolist()
numerical_columns.remove("Sample_Type")  # Exclure Sample_Type car c'est une catégorie

for col in numerical_columns:
    metadata.update_column(col, sdtype="numerical")


# Vérifier si des colonnes sont mal catégorisées
print("\nTypes de colonnes détectés par SDV :")
print(metadata.columns)

# Instancier et entraîner le modèle TVAE
model = TVAESynthesizer(
    metadata,
    enforce_min_max_values=False,  # Pour respecter les valeurs originales
    epochs=500,
    batch_size=500,
)

print("\nEntraînement en cours...")
model.fit(df_features)
print("✅ Entraînement terminé !")

# Générer de nouvelles données
synthetic_data = model.sample(num_rows=len(df_features))

# Vérifier les données générées
print("\nAperçu des données synthétiques :")
print(synthetic_data.head())

# Sauvegarder le dataset généré
synthetic_data.to_excel("dataset_synthetique_tvae.xlsx", index=False)
print("\n✅ Dataset synthétique sauvegardé sous 'dataset_synthetique_tvae.xlsx' !")
