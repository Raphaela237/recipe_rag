# RAG - Recipe Assistant with Google Gemini

![Logo du Projet](image.png)  <!-- Remplacer "image.png" par le nom de ton image -->

Ce projet propose un assistant de recette basé sur la technologie Retrieval-Augmented Generation (RAG) pour offrir une expérience interactive aux utilisateurs en leur permettant d'accéder à une liste diversifiée de recettes via des requêtes en langage naturel. L'utilisateur peut obtenir des informations sur les recettes, les instructions de préparation, et même les apports nutritionnels associés à chaque plat.

## Fonctionnalités
- **Recherche de recettes par critères** : Les utilisateurs peuvent poser des questions en langage naturel pour obtenir des recettes basées sur des critères simples (ingrédients, type de plat, etc.).
- **Instructions détaillées** : Chaque recette fournit des instructions claires et détaillées sur la préparation.
- **Informations nutritionnelles** : Pour chaque recette, des données nutritionnelles sont fournies, permettant à l'utilisateur de suivre son régime alimentaire.
- **Génération de réponses contextuelles** : Le modèle RAG combine les données des recettes et les analyse pour fournir des réponses plus précises et personnalisées.

## Installation

### Prérequis
1. **Python 3.8+**
2. Créez un environnement virtuel (optionnel mais recommandé) :
    ```bash
    python -m venv venv
    source venv/bin/activate  # Sur Windows utilisez 'venv\Scripts\activate'
    ```

3. Clonez ce projet sur votre machine locale :
    ```bash
    git clone https://github.com/Raphaela237/recipe_rag
    cd votre-depot
    ```

4. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

5. Assurez-vous de configurer les variables d'environnement en créant un fichier `.env` dans le répertoire principal avec les informations nécessaires (comme votre clé API pour Google Gemini) :
    ```bash
    cp .env.example .env  # Si tu as un fichier .env.example
    ```

6. Lancer l'application avec Streamlit :
    ```bash
    streamlit run app.py
    ```

### Variables d'Environnement
- **GEMINI_API_KEY** : Clé API pour accéder au modèle Gemini de Google.

## Utilisation

L'application Streamlit offre une interface utilisateur simple où les utilisateurs peuvent poser des questions sur des recettes en langage naturel. Le modèle génère des réponses en se basant sur un ensemble de recettes et fournit des informations sur les ingrédients, les étapes de préparation et les apports nutritionnels associés.

### Exemple d'utilisation :
- **Critères de recherche** : "I want a vegetarian recipe" ou "I want recipe without milk"
- **Réponse générée** : L'assistant renverra une recette qui correspond aux critères, accompagnée de la liste des ingrédients, des instructions de préparation, et des informations nutritionnelles.

## Structure du projet

- **app.py** : L'interface Streamlit où l'utilisateur peut poser des questions et recevoir des réponses générées par le modèle.
- **generate_embeddings.py** : Génère les embeddings des recettes et les stocke dans la base de données vectorielle Chroma
- **rag.py** : Contient les fonctions de recherche et de génération des réponses basées sur les recettes (utilisation du modèle Gemini et extraction des données pertinentes).
- **requirements.txt** : Liste des dépendances nécessaires au projet.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE).
