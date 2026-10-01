import os
import pickle
import re

import spacy
import streamlit as st

# 1. Configuration de la page (Mode moderne)
st.set_page_config(
    page_title="NLP Dashboard — Classification TALN",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé pour moderniser l'interface
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Chargement des ressources avec cache
@st.cache_resource
def charger_ressources():
    nlp = spacy.load("fr_core_news_sm")
    with open("modele_nlp.pkl", "rb") as f:
        modele = pickle.load(f)
    with open("vectoriseur_tfidf.pkl", "rb") as f:
        vectoriseur = pickle.load(f)
    return nlp, modele, vectoriseur

try:
    nlp, modele, vectoriseur = charger_ressources()
except FileNotFoundError:
    st.error("❌ Fichiers du modèle manquants. Veuillez d'abord exécuter votre script 'pipeline.py'.")
    st.stop()

# Pipeline de nettoyage TALN
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

# 3. Barre latérale (Sidebar) - Présentation du projet
with st.sidebar:
    st.image("https://icons8.com", width=80)
    st.markdown("###  Infos Candidat")
    st.markdown("**Poste visé :** Stage Data Science")
    st.markdown("**Modèle actuel :** Régression Logistique")
    st.markdown("**Précision globale :** `77%` (Baseline)")
    st.divider()
    st.markdown(" *Astuce : Ce tableau de bord permet de tester l'inférence en temps réel et d'explorer les analyses statistiques du corpus.*")

# 4. En-tête principal
st.markdown("<div class='main-title'>🤖 NLP Classifier Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Classification intelligente de textes mixtes (Français / Éwé-Mina)</div>", unsafe_allow_html=True)

# 5. Organisation du contenu en Onglets (Tabs)
onglet_demonstration, onglet_statistiques = st.tabs([" Test de l'IA en Direct", " Statistiques & Insights"])

# --- CONTENU DE L'ONGLET 1 : DÉMO EN DIRECT ---
with onglet_demonstration:
    st.subheader(" Analyse de sentiment & Catégorisation")
    st.write("Entrez une phrase combinant du français et des expressions locales togolaises pour évaluer le modèle.")
    
    phrase_utilisateur = st.text_area(
        "Saisissez votre texte ici :",
        placeholder="Exemple : C'est inadmissible, le service client est injoignable loo... Akpé pour rien !",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        bouton_analyser = st.button("Lancer l'analyse ✨", use_container_width=True, type="primary")
        
    if bouton_analyser:
        if not phrase_utilisateur.strip():
            st.warning(" Veuillez entrer du texte avant d'analyser.")
        else:
            with st.spinner(" L'IA analyse votre phrase..."):
                # Prétraitement et inférence
                phrase_nettoyee = nettoyer_texte(phrase_utilisateur)
                vecteur_numerique = vectoriseur.transform([phrase_nettoyee])
                
                prediction = modele.predict(vecteur_numerique)[0]
                probabilites = modele.predict_proba(vecteur_numerique)[0]
                classes = modele.classes_
                
                # Mise en page du résultat de façon moderne
                st.divider()
                st.markdown(f"###  Résultat de la prédiction : `{prediction}`")
                
                # Affichage des scores de confiance sous forme de barres horizontales épurées
                st.write("####  Confiance détaillée du modèle :")
                for cl, prob in zip(classes, probabilites):
                    pourcentage = prob * 100
                    st.write(f"**{cl}** : {pourcentage:.1f}%")
                    st.progress(float(prob))
                
                st.divider()
                # Encadré technique pour montrer l'arrière-boutique (Très valorisé par les recruteurs)
                st.markdown(
                    f"<div class='metric-box'>🛠️ **Pipeline TALN (Texte nettoyé transmis à l'IA) :**<br><code>{phrase_nettoyee if phrase_nettoyee else '[Texte vide après filtrage des stopwords]'}</code></div>", 
                    unsafe_allow_html=True
                )

# --- CONTENU DE L'ONGLET 2 : STATISTIQUES ---
with onglet_statistiques:
    st.subheader(" Visualisation des données du test technique")
    st.write("Voici les visuels générés automatiquement lors de la phase d'exploration du dataset de 150 lignes.")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("####  Distribution des catégories")
        if os.path.exists("distribution_classes.png"):
            st.image("distribution_classes.png", use_container_width=True)
        else:
            st.info("Le graphique 'distribution_classes.png' n'a pas encore été généré par pipeline.py.")
            
    with col_img2:
        st.markdown("####  Analyse des Nuages de Mots (Exemple)")
        # Recherche dynamique d'un nuage de mots existant dans le dossier
        fichiers_wordcloud = [f for f in os.listdir('.') if f.startswith('wordcloud_') and f.endswith('.png')]
        if fichiers_wordcloud:
            choix_wc = st.selectbox("Sélectionnez la classe à afficher :", fichiers_wordcloud)
            st.image(choix_wc, use_container_width=True)
        else:
            st.info("Aucun fichier image de type 'wordcloud_xxx.png' trouvé. Lancez 'pipeline.py'.")
