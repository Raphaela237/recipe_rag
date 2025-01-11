# importer les librairies
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import jq

# Fonction  pour extraire les métadonnées
def metadata_func(record: dict, metadata: dict) -> dict:
    """
    Met à jour le dictionnaire de métadonnées piavec des informations extraites d'un enregistrement.
    Args:
        record (dict): Enregistrement brut issu du JSON.
        metadata (dict): Métadonnées existantes.
    Returns:
        dict: Métadonnées mises à jour.
    """
    metadata["title"] = record.get("title", "")
    metadata["ingredients"] = record.get("ingredients", [])
    metadata["calories"] = record.get("calories")
    metadata.pop("source")
    metadata.pop("seq_num")
    return metadata

# Étape 1 : Charger le fichier JSON avec jq_schema
file_path = "recipe.json"
jq_schema = ".[]"  # Parcourt chaque élément du tableau JSON

loader = JSONLoader(
    file_path=file_path,
    jq_schema=jq_schema,  # Schéma jq pour extraire les objets
    content_key="directions",  # Clé contenant le contenu principal
    metadata_func=metadata_func  # Fonction pour gérer les métadonnées
)

# Charger les données
documents = loader.load()

# Étape 2 : Diviser les données en morceaux (chunks)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Taille maximale des chunks
    chunk_overlap=50,  # Chevauchement pour garder le contexte
    separators=["\n", " ", ""]  # Séparateurs à utiliser
)

# Diviser les documents
chunks = splitter.split_documents(documents)

embedding_function = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2", model_kwargs = {'device':'cpu'})


vectorstore = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_db_nccn")
print(vectorstore._collection.count())

