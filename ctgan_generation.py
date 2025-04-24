import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import torch

print("🚀 GPU disponible :", torch.cuda.is_available())
print("🔍 Appareil utilisé par PyTorch :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Charger le dataset
file_path = "dataset_reduit_et_normaliser.xlsx"
df = pd.read_excel(file_path)

# Créer les métadonnées
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Définir le type correct pour la colonne cible
metadata.update_column("Sample_Type", sdtype="categorical")

# Forcer toutes les autres colonnes à être numériques
for col in df.columns:
    if col != "Sample_Type":
        metadata.update_column(col, sdtype="numerical")

# Vérification des types
print("\n✅ Types de colonnes configurés :")
for col, meta in metadata.columns.items():
    print(f" - {col} : {meta['sdtype']}")

# Créer et entraîner le modèle CTGAN
model = CTGANSynthesizer(
    metadata,
    epochs=500,
    batch_size=500
)

print("\n🚧 Entraînement du modèle CTGAN...")
model.fit(df)
print("✅ Entraînement terminé !")

# Génération
synthetic_data = model.sample(num_rows=len(df))

# Affichage d’un aperçu
print("\n🔍 Aperçu des données synthétiques générées :")
print(synthetic_data.head())

# Sauvegarde
synthetic_data.to_excel("dataset_synthetique_ctgan.xlsx", index=False)
print("\n✅ Dataset synthétique sauvegardé sous 'dataset_synthetique_ctgan.xlsx'")