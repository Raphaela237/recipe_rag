import os
import streamlit as st
from rag import get_relevant_context_from_db, generate_rag_prompt, answer

# Charger la clé API Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Si ce n'est pas déjà dans l'état de la session, initialise les réponses, requêtes et historique
if 'history' not in st.session_state:
    st.session_state.history = []

# Définir l'interface de l'application Streamlit
st.title("Healthy Cooking AI Chatbot")
st.subheader("What do you want?")

# Conteneur pour afficher la conversation
response_container = st.container()

# Zone de saisie de la question
prompt = st.chat_input("Your question:")

if prompt:
    # Ajouter la question de l'utilisateur à l'historique
    st.session_state.history.append({
        'role': 'user',
        'content': prompt
    })

    # Afficher le message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('💡 Typing...'):
        # Récupérer le contexte pertinent depuis la base de données
        context = get_relevant_context_from_db(prompt)

        # Générer le prompt avec la question, le contexte et l'historique
        generated_prompt = generate_rag_prompt(query=prompt, context=context, history=[msg['content'] for msg in st.session_state.history])

        # Appeler la fonction pour obtenir la réponse via l'API Gemini
        answer_text = answer(generated_prompt)  # Utilisation de la fonction answer

        # Ajouter la réponse de l'assistant à l'historique
        st.session_state.history.append({
            'role': 'assistant',
            'content': answer_text
        })

    # Afficher l'historique de la conversation
    with response_container:
        for msg in st.session_state.history:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])
