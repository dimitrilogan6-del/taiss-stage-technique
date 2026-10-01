import pickle
import re

import spacy
import streamlit as st

# Configuration de la page Streamlit
st.set_page_config(page_title="Démo NLP — Stage Data Science", page_icon="🤖", layout="centered")

st.title("🤖 Analyseur de Commentaires Mixtes (FR / Éwé-Mina)")
st.write("Cette interface permet de tester en direct le modèle de classification entraîné sur les 150 lignes du dataset.")

# 1. Chargement des outils nécessaires (Modèle, Vectoriseur et spaCy)
@st.cache_resource
def charger_ressources():
    nlp = spacy.load("fr_core_news_sm")
    # Chargement du modèle et du vectoriseur sauvegardés par votre pipeline.py
    with open("modele_nlp.pkl", "rb") as f:
        modele = pickle.load(f)
    with open("vectoriseur_tfidf.pkl", "rb") as f:
        vectoriseur = pickle.load(f)
    return nlp, modele, vectoriseur

try:
    nlp, modele, vectoriseur = charger_ressources()
    st.success("✅ Modèle et dictionnaire linguistique chargés avec succès !")
except FileNotFoundError:
    st.error("❌ Fichiers de modèle manquants. Veuillez d'abord exécuter votre script `pipeline.py` pour générer les fichiers `.pkl`.")
    st.stop()

# 2. Fonction de nettoyage identique au Notebook
stopwords_ewe_mina = {"la", "loo", "o", "ao", "yea", "yé", "ke", "na", "nu", "a", "xo"}
stopwords_francais = nlp.Defaults.stop_words

def nettoyer_texte(texte_brut):
    if not isinstance(texte_brut, str):
        return ""
    texte = texte_brut.lower()
    texte = re.sub(re.compile(r'[^\w\s]', re.UNICODE), ' ', texte)
    texte = re.sub(r'\d+', ' ', texte)
    doc = nlp(texte)
    mots_filtres = []
    for token in doc:
        mot = token.text.strip()
        lemme = token.lemma_
        if not mot or len(mot) < 2:
            continue
        if len(lemme) > 1 and lemme not in stopwords_francais and mot not in stopwords_ewe_mina:
            mots_filtres.append(lemme)
    return " ".join(mots_filtres)

# 3. Zone de saisie utilisateur
st.subheader("📝 Saisissez un commentaire à analyser :")
phrase_utilisateur = st.text_input("Exemple : Le produit est très bien, akpé loo !", "")

if st.button("Analyser le texte 🚀"):
    if phrase_utilisateur.strip() == "":
        st.warning("Veuillez entrer du texte avant de cliquer sur le bouton.")
    else:
        # Prétraitement de l'entrée
        phrase_nettoyee = nettoyer_texte(phrase_utilisateur)
        
        # Vectorisation
        vecteur_numerique = vectoriseur.transform([phrase_nettoyee])
        
        # Prédiction et probabilités
        prediction = modele.predict(vecteur_numerique)[0]
        probabilites = modele.predict_proba(vecteur_numerique)[0]
        classes = modele.classes_
        
        # Affichage des résultats
        st.markdown(f"### 🎉 Catégorie prédite : **`{prediction}`**")
        
        # Affichage des scores de confiance
        st.write("📊 **Confiance du modèle :**")
        for cl, prob in zip(classes, probabilites):
            st.write(f"- **{cl}** : {prob*100:.1f}%")
            st.progress(float(prob))
            
        st.info(f"**Texte après nettoyage TALN :** `{phrase_nettoyee}`")
