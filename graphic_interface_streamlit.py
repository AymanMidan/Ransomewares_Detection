import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.utils import shuffle
import openpyxl

# ---- FICHIERS ----
dataset_paths = {
    "Original": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier.csv",
    "CouplaGAN": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/dataset_synthetique_copulagan2.csv",
    "CTGAN_100K_1":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_1.csv",
    "CTGAN_100K_2":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_2.csv",
    "CTGAN_100K_3":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_3.csv",
    "CTGAN_100K_4":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_4.csv",
    "CTGAN_100K_5":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_5.csv",
    "CTGAN_100K_6":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_6.csv",
    "CTGAN_100K_7":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_7.csv",
    "CTGAN_100K_8":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_8.csv",
    "CTGAN_100K_9":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_9.csv",
    "CTGAN_100K_10":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_10.csv",
    
    
    "TVAE": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/dataset_synthetique_tvae%20(1).csv",
    "CTGAN": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/dataset_synthetique_ctgan.csv"
}

def load_ctgan_100k():
    """Charge les 10 parties du CTGAN 100K"""
    return pd.concat([pd.read_csv(dataset_paths[f'CTGAN_100K_{i+1}']) for i in range(10)], ignore_index=True)


# ---- FONCTION DE LECTURE ----
def lire_fichier(chemin):
    if chemin.endswith('.csv'):
        return pd.read_csv(chemin)
    elif chemin.endswith('.xlsx'):
        return pd.read_excel(chemin)

# ---- MODELE ----
def create_model(model_name):
    if model_name == 'XGBoost':
        return xgb.XGBClassifier()
    elif model_name == 'RF':
        return RandomForestClassifier()
    elif model_name == 'SVM':
        return SVC(probability=True)
    elif model_name == 'MLP':
        return MLPClassifier()
    elif model_name == 'Stack(XGBoost + SVM + MLP)':
        return VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()),
            ('svm', SVC(probability=True)),
            ('mlp', MLPClassifier())
        ], voting='soft')
    elif model_name == 'Stack(XGBoost + RF + MLP)':
        return VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()),
            ('rf', RandomForestClassifier()),
            ('mlp', MLPClassifier())
        ], voting='soft')
def shapiro(df):
    from scipy.stats import shapiro
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
     st.write(f"{col[:25]:<25} | p-value = {p:.4f} | {'Normal' if p > 0.05 else 'Non-normal'}")

     if p > 0.05:
        normal_count += 1

# 4. Recommandation finale
     st.write("\n=== Résumé ===")
     st.write(f"Colonnes normales    : {normal_count}/100")
     st.write(f"Colonnes non-normales: {100 - normal_count}/100")

     if normal_count >= 70:
        st.write("\nRecommandation : StandardScaler (majorité normale)")
     elif normal_count >= 30:
        st.write("\nRecommandation : PowerTransformer (mixte normal/non-normal)")
     else:
        st.write("\nRecommandation : RobustScaler (majorité non-normale)")

    
    

    # Normalisation des donnéesimport pandas as pd
from sklearn.manifold import TSNE
    

    # Demander à l'utilisateur le chemin du fichier

def tsne(df):
    
    y = df["Sample_Type"]
    X = df.drop("Sample_Type", axis=1)

    # On garde que les colonnes numériques pour t-SNE
    X_numeric = X.select_dtypes(include='number')

    # Projection t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_model.fit_transform(X_numeric)

    # Couleurs personnalisées
    colors = {1: "red", 0: "blue"}
    point_colors = y.map(colors)

    # Affichage
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
    plt.title("Projection t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    st.pyplot(fig)


    # t-SNE



# ---- SIDEBAR ----
st.sidebar.title("📂 Sélection du Dataset")
if "datasets_choisis" not in st.session_state:
    st.session_state.datasets_choisis = []

dataset_choice = st.sidebar.multiselect(
    "Choisissez un ou plusieurs datasets",
   options=['Original','CTGAN_100k','CTGAN','TVAE','CouplaGAN'],default=['Original']
)   
st.session_state.datasets_choisis = dataset_choice
df = pd.DataFrame()

if dataset_choice:
    all_data = []
    
    for name in dataset_choice:
        if name == 'CTGAN_100k':
            all_data.append(load_ctgan_100k())
        elif name in dataset_paths:
            all_data.append(pd.read_csv(dataset_paths[name]))
    
    if all_data:
        df = pd.concat(all_data, axis=0)
        df = shuffle(df, random_state=42)
    str=""
    j=0
    for i in range(len(dataset_choice)):                                                                                                                    
        j+=1
        str=str+dataset_choice[i]
        if j<len(dataset_choice):
              str=str+"+"
        
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
st.session_state.df=df




if "go_data" not in st.session_state:
    st.session_state.go_data = False
go_data = st.sidebar.button("🚀 Go (Charger Dataset)")
# ---- PRINCIPAL ----
st.title("Ransomeware vs Goodware")

if dataset_choice:
  if go_data:
    dtt=df
    dtt['Sample_Type']=dtt['Sample_Type'].replace({0:"G",1:"R"})
    st.sidebar.write("Chargement: ",str)
    
    st.dataframe(dtt.head(11))
    dtt['Sample_Type']=dtt['Sample_Type'].replace({"G":0,"R":1})

    st.session_state.go_data=True

else:
      
      st.sidebar.markdown("pas de choix de dataset")
if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
            tsne(st.session_state.df)

# ---- SELECTION MODELE ----
st.sidebar.write("---")
choix_models = st.sidebar.multiselect(
    "🛠️ Choisissez les modèles à entraîner",
    options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
    default=["XGBoost"]
)
go_model = st.sidebar.button("🚀 Go (Exécuter Modèle)")
if dataset_choice:
  df=st.session_state.df
  X = df.drop('Sample_Type', axis=1)
  y = df['Sample_Type']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if 'go_model' not in st.session_state:
    st.session_state.go_model=False
if go_model:
 if dataset_choice:
    st.session_state.go_model=True

    str=""
    j=0
    for i in range(len(choix_models)):
        j+=1
        str=str+choix_models[i]
        if j<len(choix_models):
              str=str+" et "
    st.sidebar.write("Chargement :",str)

    for name in choix_models:
        model = create_model(name)
        with st.spinner(f"⏳ Entraînement du modèle {name} en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        st.title(f"---\n### 🔍 Résultats pour le modèle : `{name}`")
        st.markdown(f"""
        ✅ **Accuracy**        : `{acc:.4f}`  
        ✅ **Recall Score**    : `{recall:.4f}`  
        ✅ **Precision Score** : `{precision:.4f}`  
        ✅ **F1 Score**        : `{f1:.4f}`  
        ✅ **AUC-ROC**         : `{auc}`
        """)

        with st.expander("📋 Rapport détaillé"):
            rapport = classification_report(y_test, y_pred,target_names=["G", "R"], output_dict=False)
            st.text(rapport)
            st.markdown("### 🧾 Matrice de confusion")

            with st.spinner(f"⏳ Entraînement du modèle {name} en cours..."):
                 cm = confusion_matrix(y_test, y_pred)
                 fig_cm, ax_cm = plt.subplots()
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["G", "R"])
                 disp.plot(ax=ax_cm, cmap="Blues")
                 st.pyplot(fig_cm)
 else:
     st.markdown("pas de chargement de dataset")

         
if st.session_state.go_model:
 if st.sidebar.button("🚀 courbe d'apprentissage"):
        for name in choix_models:
         model = create_model(name)
         st.markdown("### 🎓 Courbe d'apprentissage")

         with st.spinner(f"⏳ Courbe d'apprentissage{name} en cours..."):
          
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, scoring='accuracy',
                train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_mean, label="Train")
            ax.plot(train_sizes, test_mean, label="Validation")
            ax.set_title("Learning Curve")
            ax.set_xlabel("Training Size")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)
if st.session_state.go_model:
 if st.sidebar.button("🚀 ROC curve"):
        for name in choix_models:
         model = create_model(name)
         st.markdown("### ROC curve")

         with st.spinner(f"⏳ ROC Curve {name} en cours..."):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

 
            
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc, ax_roc = plt.subplots()
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

                ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('Faux positifs')

                ax_roc.set_ylabel('Vrais positifs')
                ax_roc.legend()
                st.pyplot(fig_roc)
        
