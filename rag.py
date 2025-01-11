
import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Configurer l'API de Gemini
genai.configure(api_key=gemini_api_key)

# Fonction pour obtenir un contexte pertinent depuis la base de données
def get_relevant_context_from_db(query):
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    context = "" 
    search_results = vector_db.similarity_search(query, k=8)
    for result in search_results:
        context += result.page_content + "\n"
    return context

# Fonction pour générer le prompt de RAG (retrieval-augmented generation)
def generate_rag_prompt(query, context, history):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
    history_text = "\n".join(history)
    prompt = f"""
              You are an AI assistant designed to help users find healthy recipes based on the provided database. Your task is to answer questions regarding recipes, including their ingredients, preparation steps, and nutritional information. You must only use information available from the provided recipe database.

Instructions:

Answer only with details found in the database: When responding, make sure to only use information present in the provided documents. Do not generate answers based on your internal knowledge or assumptions.
Be sure to consider the season, ambiance, and dietary restrictions mentioned in the query.
If the user requests a recipe for a specific occasion (e.g., romantic, summer), focus on those themes and suggest ingredients that match the request.
If the user specifies a restriction (e.g., "without fish", "without meat"), make sure to exclude those ingredients from the recipe suggestions.
Always offer structured and clear responses, including recipe names, ingredients, preparation steps, and nutritional information if available on documents.
Extract and apply constraints mentioned in the user query. For example, avoid specific ingredients or types of recipes if specified.
Refer to the conversation history to maintain coherence in your responses. If the user's question refers to previous exchanges (e.g., a recipe already provided), use that context to answer.
Avoid providing sensitive, inappropriate, or potentially incorrect information: Your responses should always be appropriate, relevant, and accurate based on the data in the database. If any information appears unclear or unverified, avoid providing an answer.
If the answer is not in the database: If the question is outside the scope of the provided documents, you must clearly state: "I do not have information to answer this question based on the available documents."
When answering: Make sure to offer clear and structured responses, including recipe names, ingredients, preparation steps, and calorie details if requested. If any part of the recipe information is missing, mention that clearly.
Maintain a friendly but professional tone: Your tone should always be warm, respectful, and engaging while maintaining professionalism. Strive to provide helpful responses that make users feel comfortable and confident in their search for healthy recipes.
              Conversation history:
              {history_text}

              QUESTION: {query}
              CONTEXT: {context}

              ANSWER:
              """
    return prompt


# Fonction qui configure l'API, génère la réponse du modèle et la retourne
def answer(prompt):
    # Configurer l'API avec la clé
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name="gemini-pro")
    response = model.generate_content(prompt)
    return response.text
