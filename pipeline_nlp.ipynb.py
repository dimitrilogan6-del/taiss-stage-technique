# =====================================================================
# CELLULE 1 : Importation des bibliothèques et configuration
# =====================================================================
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Configuration des graphiques
sns.set_theme(style="whitegrid")

# =====================================================================
# CELLULE 2 : 1.1 Chargement et Exploration Globale
# =====================================================================
nom_fichier = "dataset_nlp_test_tal-Excel.csv"

# Lecture du fichier avec détection du séparateur Excel (souvent ';' en français)
try:
    df = pd.read_csv(nom_fichier, sep=';')
    if 'texte' not in df.columns:
        df = pd.read_csv(nom_fichier, sep=',')
except Exception:
    df = pd.read_csv(nom_fichier, sep=',')

print(f"=== Structure du Dataset ===")
print(f"Nombre de lignes et colonnes (Shape) : {df.shape}")
print(f"\n=== Types des données ===\n{df.dtypes}")
print(f"\n=== Aperçu des 3 premières lignes ===\n")
print(df.head(3))

# Visualisation de la distribution des classes
plt.title("Distribution des classes au sein du Dataset")
plt.xlabel("Catégories")
plt.ylabel("Nombre de documents")
plt.tight_layout()
plt.savefig("distribution_classes.png") # <--- Enregistre l'image automatiquement
plt.close() # <--- Ferme le graphique en arrière-plan sans bloquer


# Calcul de la longueur moyenne des textes par catégorie
df['longueur_mots'] = df['texte'].apply(lambda x: len(str(x).split()))
longueur_moyenne = df.groupby('categorie')['longueur_mots'].mean().reset_index()
print(f"\n=== Longueur moyenne des textes par catégorie (en mots) ===")
print(longueur_moyenne.to_string(index=False))

# =====================================================================
# CELLULE 3 : 1.2 Nettoyage avancé (Français & Éwé/Mina) & 1.3 Visualisations
# =====================================================================
# Chargement du modèle français de spaCy
nlp = spacy.load("fr_core_news_sm")

# Stratégie linguistique pour les langues locales (Togo)
stopwords_ewe_mina = {"la", "loo", "o", "ao", "yea", "yé", "ke", "na", "nu", "a", "xo"}
stopwords_francais = nlp.Defaults.stop_words

def pipeline_nettoyage_tal(texte_brut):
    if not isinstance(texte_brut, str):
        return ""
    
    # Passage en minuscules
    texte = texte_brut.lower()
    
    # Suppression de la ponctuation, des caractères spéciaux et des chiffres
    texte = re.sub(re.compile(r'[^\w\s]', re.UNICODE), ' ', texte)
    texte = re.sub(r'\d+', ' ', texte)
    
    # Tokenisation et analyse morphologique
    doc = nlp(texte)
    mots_filtres = []
    
    for token in doc:
        mot = token.text.strip()
        lemme = token.lemma_
        
        # Ignorer les chaînes vides ou lettres isolées
        if not mot or len(mot) < 2:
            continue
            
        # Exclusion des stopwords français et des particules vides éwé/mina
        if lemme not in stopwords_francais and mot not in stopwords_ewe_mina:
            mots_filtres.append(lemme)
            
    return " ".join(mots_filtres)

print("🧹 Application du traitement linguistique sur les 150 lignes...")
df['texte_nettoye'] = df['texte'].apply(pipeline_nettoyage_tal)
print("✅ Nettoyage terminé.")

# 1.3 Génération des WordClouds et des Top 15 termes par catégorie
categories = df['categorie'].unique()

for cat in categories:
    df_cat = df[df['categorie'] == cat]
    texte_global = " ".join(df_cat['texte_nettoye'])
    liste_mots = texte_global.split()
    
    # Nuage de mots
    if texte_global.strip():
        wordcloud = WordCloud(width=700, height=400, background_color='white', colormap='plasma').generate(texte_global)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Nuage de mots de la catégorie : {cat.upper()}", fontsize=14, fontweight='bold')
        plt.savefig(f"wordcloud_{cat}.png") # <--- Crée une image par catégorie (ex: wordcloud_positif.png)
        plt.close() # <--- Libère la mémoire et passe à la suite

    
    # Top 15 termes
    top_15 = Counter(liste_mots).most_common(15)
    print(f"\n📊 Top 15 des termes les plus fréquents pour la catégorie [{cat.upper()}] :")
    for mot, freq in top_15:
        print(f"   - '{mot}' : apparu {freq} fois")
    print("-" * 60)

# =====================================================================
# CELLULE 4 : 2.1 Vectorisation TF-IDF et Division Train/Test
# =====================================================================
# Instanciation du vectoriseur statistique
vectoriseur = TfidfVectorizer()

# Séparation des données (80% entraînement, 20% validation) avec stratification
X_train_brut, X_test_brut, y_train, y_test = train_test_split(
    df['texte_nettoye'], 
    df['categorie'],
    test_size=0.2, 
    random_state=42, 
    stratify=df['categorie']
)

# Transformation numérique des données textuelles
X_train = vectoriseur.fit_transform(X_train_brut)
X_test = vectoriseur.transform(X_test_brut)

print(f"Matrice d'entraînement numérique générée : {X_train.shape}")
print(f"Matrice de test numérique générée : {X_test.shape}")

# =====================================================================
# CELLULE 5 : 2.2 Entraînement des algorithmes de classification
# =====================================================================
print("\n⚙️ Entraînement des modèles de classification en cours...")

# Algorithme A : Régression Logistique (Baseline linéaire classique)
model_lr = LogisticRegression(class_weight='balanced', random_state=42)
model_lr.fit(X_train, y_train)

# Algorithme B : Random Forest (Modèle d'ensemble non linéaire)
model_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)

print("✅ Entraînements finalisés.")

# =====================================================================
# CELLULE 6 : 2.3 Évaluation des performances et Analyse des erreurs
# =====================================================================
# Prédictions
y_pred_lr = model_lr.predict(X_test)
y_pred_rf = model_rf.predict(X_test)

# Calcul des indicateurs clés demandés
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='macro')

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')

# Tableau de synthèse comparatif
tableau_synthese = pd.DataFrame({
    'Algorithme de Classification': ['Régression Logistique', 'Random Forest'],
    'Accuracy (Précision globale)': [f"{acc_lr:.2f}", f"{acc_rf:.2f}"],
    'F1-Score Macro': [f"{f1_lr:.2f}", f"{f1_rf:.2f}"]
})

print("\n📊 === TABLEAU DE SYNTHÈSE DES MODÈLES ===")
print(tableau_synthese.to_string(index=False))

print("\n🧩 === MATRICE DE CONFUSION (Régression Logistique) ===")
print(confusion_matrix(y_test, y_pred_lr))
print("\n🔬 Rapport détaillé de classification (Régression Logistique) :")
print(classification_report(y_test, y_pred_lr))

# Extraction et analyse automatique des exemples mal classés
indices_test = y_test.index
analyse_erreurs_df = pd.DataFrame({
    'Texte d\'origine brut': df.loc[indices_test, 'texte'],
    'Texte après nettoyage': X_test_brut,
    'Vraie Catégorie (Ground Truth)': y_test,
    'Catégorie Prédite par l\'IA': y_pred_lr
})

# Isolation des lignes où la prédiction est fausse
exemples_mal_classes = analyse_erreurs_df[
    analyse_erreurs_df['Vraie Catégorie (Ground Truth)'] != analyse_erreurs_df['Catégorie Prédite par l\'IA']
]

print(f"\n🔍 === ANALYSE DES EXEMPLES MAL CLASSÉS ({len(exemples_mal_classes)} erreurs trouvées) ===")
if len(exemples_mal_classes) > 0:
    # On affiche jusqu'à 5 exemples d'erreurs pour l'évaluation humaine
    print(exemples_mal_classes.head(5).to_string(index=False))
else:
    print("Aucune erreur détectée sur l'échantillon de test.")
