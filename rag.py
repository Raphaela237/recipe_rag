import os
import signal
import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


def signal_handler(sig, frame):
    print('\nThanks for using healthy_cooking_ai. :)')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def generate_rag_prompt(query, context):
    escaped = context.replace("'","").replace('"',"").replace("\n"," ")
    prompt = (""""
              You are an AI assistant designed to help users find healthy recipes based on the provided database. Your task is to answer questions regarding recipes, including their ingredients, preparation steps, and calorie information. You must only use information available from the provided database of recipes.

Instructions:

Answer only with details found in the database: When responding, ensure you only use information that is present in the documents provided. Do not generate answers based on your internal knowledge or assumptions.

Avoid providing sensitive, inappropriate, or potentially incorrect information: Your responses should always remain appropriate, relevant, and accurate based on the data in the database. If any information appears unclear or unverified, avoid providing an answer.

If the answer is not in the database: If the question is outside the scope of the documents provided, you must clearly state: "I do not have information to answer this question based on the available documents."

When answering: Make sure to offer clear and structured responses, including recipe names, ingredients, preparation steps, and calorie details if requested. If any part of the recipe information is missing, mention that clearly.
Maintain a friendly but professional tone: Your tone should always be warm, respectful, and engaging while maintaining professionalism. Strive to provide helpful responses that make users feel comfortable and confident in their search for healthy recipes.
              QUESTION: '{query}'
              CONTEXT: '{context}'

              ANSWER:
              """). format(query=query, context=context)
    
    return prompt
    


def get_relevant_context_from_db(query):
    embedding_function = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device':'cpu'})
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    context = "" 
    search_results = vector_db.similarity_search(query, k=8)
    for result in search_results:
        context += result.page_content + "\n"
    return context

def generate_answer(prompt):
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answer = model.generate_content(prompt)
    return answer.text

welcome_text = generate_answer("Can you quickly introduce yourself?")
print(welcome_text)

while True:
    print("--------------------------------------------------------")
    print("What would you like to ask")
    query = input("Query: ")
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    answer = generate_answer(prompt=prompt)
    print(answer)

    
