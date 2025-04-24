import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Fonction pour charger, séparer, et appliquer t-SNE
def compute_tsne(file_path, label_column='Sample_Type'):
    df = pd.read_excel(file_path)
    X = df.drop(columns=[label_column])
    y = df[label_column]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["Sample_Type"] = y
    return df_tsne

# Charger et transformer les deux datasets
tsne_original = compute_tsne("dataset_reduit_et_normaliser.xlsx")
tsne_ctgan = compute_tsne("dataset_synthetique_copulagan2.xlsx")

# Affichage côte à côte
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Dataset Original
sns.scatterplot(
    data=tsne_original,
    x="TSNE1", y="TSNE2",
    hue="Sample_Type",
    palette={0: "blue", 1: "red"},
    alpha=0.7,
    ax=axes[0]
)
axes[0].set_title("t-SNE Dataset Original")
axes[0].legend(title="Classe", labels=["Goodware (0)", "Ransomware (1)"])

# Dataset CTGAN
sns.scatterplot(
    data=tsne_ctgan,
    x="TSNE1", y="TSNE2",
    hue="Sample_Type",
    palette={0: "blue", 1: "red"},
    alpha=0.7,
    ax=axes[1]
)
axes[1].set_title("t-SNE Dataset Synthétique (CopulaGAN)")
axes[1].legend(title="Classe", labels=["Goodware (0)", "Ransomware (1)"])

plt.tight_layout()
plt.show()
