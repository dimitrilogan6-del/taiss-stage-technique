#  Analyse de Textes Mixtes (Français / Éwé-Mina) — Pipeline TALN(TRAITEMENT AUTOMATIQUE DES LANGAGES NATURELS)
Ce projet présente un pipeline complet de Traitement Automatique du Langage Naturel (NLP / TALN) développé en Python pour explorer, nettoyer, vectoriser et classifier un corpus de 150 lignes incluant des expressions locales du Togo (Éwé/Mina).
##  Approche & Choix Techniques
1. **Prétraitement & Stratégie Linguistique (spaCy & Regex)** :
   - Nettoyage via expressions régulières (minuscules, suppression des chiffres, ponctuation).
   - **Stopwords Français** : Élimination via `spaCy` pour réduire le bruit sémantique.
   - **Gestion de l'Éwé/Mina** : Dictionnaire d'exclusion ciblé pour filtrer les particules orales vides (*loo, ke, yé, na*) tout en préservant les racines lexicales significatives (*Akpé, Mawu*).
2. **Vectorisation (TF-IDF)** :
   - Privilégié face au `CountVectorizer` pour pénaliser les termes omniprésents et valoriser les mots rares et discriminants sur un petit corpus.
3. **Modélisation & Robustesse** :
   - Division stratifiée (80% Train / 20% Test, `random_state=42`).
   - Comparaison entre **Régression Logistique** et **Random Forest** avec le paramètre `class_weight='balanced'`.

##  Résultats & Synthèse
La **Régression Logistique** surpasse le Random Forest avec une **Précision globale de 77%**. Les erreurs proviennent principalement des phrases trop courtes ou contenant des négations directes.

##  Installation & Utilisation
### Prérequis
```bash
pip install pandas spacy scikit-learn openpyxl matplotlib seaborn wordcloud streamlit
python -m spacy download fr_core_news_sm
```
### Exécution du Pipeline
```bash
python pipeline_nlp.ipynb.py
```
### Lancement de l'Interface Graphique (Bonus )
```bash
streamlit run app.py
```
## Limites & Pistes d'Amélioration
- **Limites** : Perte de contexte avec le sac de mots (Bag of Words) et sensibilité aux mots hors vocabulaire (OOH).
- **Pistes** : Utilisation de N-grammes ou fine-tuning de Transformers multilingues (mBERT / XLM-RoBERTa).
