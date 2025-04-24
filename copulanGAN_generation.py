import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer
import torch

# ✅ Vérification GPU
print("🚀 GPU disponible :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("🖥️ Appareil utilisé :", torch.cuda.get_device_name(0))
else:
    print("🖥️ Appareil utilisé : CPU")

# 📥 Charger le dataset
file_path = "dataset_reduit_et_normaliser.xlsx"
df = pd.read_excel(file_path)

# 🔧 Créer les métadonnées
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# 🎯 Mettre "Sample_Type" en catégorielle
metadata.update_column("Sample_Type", sdtype="categorical")

# 📊 Forcer toutes les autres colonnes à être numériques
for col in df.columns:
    if col != "Sample_Type":
        metadata.update_column(col, sdtype="numerical")

# ✅ Afficher les types détectés
print("\n📌 Types de colonnes :")
for col, meta in metadata.columns.items():
    print(f" - {col}: {meta['sdtype']}")

# 🚀 Créer et entraîner le modèle CopulaGAN
model = CopulaGANSynthesizer(
    metadata,
    epochs=300,  # Augmenter légèrement pour un meilleur apprentissage
    batch_size=500,  # Batch size moyen pour plus de stabilité
    generator_dim=(256, 256),  # Capacité légèrement augmentée
    discriminator_dim=(256, 256),
    generator_lr=2e-4,  # Learning rate spécifié
    discriminator_lr=2e-4,
    cuda=True,
    verbose=True  # Pour suivre la progression
)

print("\n⏳ Entraînement du modèle CopulaGAN...")
model.fit(df)
print("✅ Entraînement terminé !")

# 🧪 Générer les données synthétiques
synthetic_data = model.sample(num_rows=len(df))

# 💾 Sauvegarder le dataset synthétique
output_path = "dataset_synthetique_copulagan2.xlsx"
synthetic_data.to_excel(output_path, index=False)
print(f"\n✅ Dataset synthétique sauvegardé dans : {output_path}")

