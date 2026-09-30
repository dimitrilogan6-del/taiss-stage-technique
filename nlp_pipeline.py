import os
import pickle

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

print(" Initialisation du pipeline NLP pour mon stage...")

# 1. Chargement du modèle spaCy français
nlp = spacy.load("fr_core_news_sm")

def nettoyer_texte(texte_brut):
    if not isinstance(texte_brut, str):
        return ""
    doc = nlp(texte_brut.lower())
    mots_nettoyes = [token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    return " ".join(mots_nettoyes)

# 2. Détection et chargement de votre fichier exact
nom_base = "dataset_nlp_test_tal-Excel"
df = None

# Résolution des chemins selon votre terminal PowerShell
if os.path.exists(f"{nom_base}.csv"):
    chemin_fichier = f"{nom_base}.csv"
elif os.path.exists(nom_base):
    chemin_fichier = nom_base
else:
    # Si PowerShell est déjà à l'intérieur du dossier portant le nom du fichier
    chemin_fichier = "dataset_nlp_test_tal-Excel.csv"

try:
    df = pd.read_csv(chemin_fichier, sep=';')
    if 'texte' not in df.columns:
        df = pd.read_csv(chemin_fichier, sep=',')
except Exception:
    df = pd.read_csv(chemin_fichier, sep=',')

print(f"✅ Fichier chargé avec succès !")
print(f"📊 Nombre de lignes détectées : {len(df)}")

# 3. Traitement NLP sur la colonne correcte : 'texte'
print("🧹 Nettoyage des 150 phrases en cours (lemmatisation)...")
df['text_nettoye'] = df['texte'].apply(nettoyer_texte)

# 4. Séparation Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    df['text_nettoye'], 
    df['categorie'],
    test_size=0.2, 
    random_state=42,
    stratify=df['categorie']
)

# 5. Vectorisation
print(" Vectorisation des textes...")
vectoriseur = TfidfVectorizer()
X_train_veto = vectoriseur.fit_transform(X_train)
X_test_veto = vectoriseur.transform(X_test)

# 6. Modèle
print(" Entraînement de la Régression Logistique...")
modele = LogisticRegression(class_weight='balanced')
modele.fit(X_train_veto, y_train)

# 7. Évaluation
y_pred = modele.predict(X_test_veto)
print("\n --- RÉSULTATS DU MODÈLE SUR LE TEST SET ---")
print(f"Précision globale (Accuracy) : {accuracy_score(y_test, y_pred):.2f}")
print("\nRapport détaillé par catégorie :")
print(classification_report(y_test, y_pred))


print("\n Sauvegarde du modèle et du vectoriseur...")
# Sauvegarde du modèle de Machine Learning
with open("modele_nlp.pkl", "wb") as f:
    pickle.dump(modele, f)

# Sauvegarde du vectoriseur TF-IDF (indispensable pour transformer les futurs textes)
with open("vectoriseur_tfidf.pkl", "wb") as f:
    pickle.dump(vectoriseur, f)

print("✅ Fichiers 'modele_nlp.pkl' et 'vectoriseur_tfidf.pkl' sauvegardés avec succès !")
