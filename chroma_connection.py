import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# --- ChromaDB Persistent Client Setup ---

# IMPORTANT: The path below is where your ChromaDB data will be stored.
# This ensures that the data persists even after the script has stopped running.
DB_PATH = "chroma_db_data"

def create_chroma_client(path=DB_PATH):
    """
    Creates a persistent ChromaDB client.
    
    Args:
        path (str): The directory where the database files will be stored.
    
    Returns:
        chromadb.PersistentClient: An instance of the persistent client.
    """
    try:
        # Create a client that points to a persistent directory
        client = chromadb.PersistentClient(path=path)
        print(f"ChromaDB client created successfully at: {path}")
        return client
    except Exception as e:
        print(f"Error creating ChromaDB client: {e}")
        return None

def get_or_create_collection(client, collection_name="chat_history"):
    """
    Gets or creates a ChromaDB collection for storing chat documents.
    
    Args:
        client (chromadb.PersistentClient): The ChromaDB client instance.
        collection_name (str): The name of the collection.
        
    Returns:
        chromadb.Collection: The collection instance.
    """
    try:
        # Get or create the collection. This is a safe operation.
        collection = client.get_or_create_collection(
            name=collection_name,
            # The default embedding function is great for general use cases.
            embedding_function=DefaultEmbeddingFunction()
        )
        print(f"Collection '{collection_name}' ready.")
        return collection
    except Exception as e:
        print(f"Error getting/creating collection: {e}")
        return None

def store_chat_message(collection, message_text, message_id):
    """
    Stores a chat message as a document in the ChromaDB collection.
    
    Args:
        collection (chromadb.Collection): The collection to add the document to.
        message_text (str): The content of the chat message.
        message_id (str): A unique ID for the message.
    """
    try:
        collection.add(
            documents=[message_text],
            metadatas=[{"source": "chat_log"}],  # Optional metadata
            ids=[message_id]
        )
        print(f"Successfully stored message with ID: {message_id}")
    except Exception as e:
        print(f"Error storing message: {e}")

def retrieve_chat_messages(collection, query_text):
    """
    Retrieves chat messages similar to the query text from the collection.
    
    Args:
        collection (chromadb.Collection): The collection to query.
        query_text (str): The query text to find similar messages for.
        
    Returns:
        dict: The query results.
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=2 # Retrieve the top 2 most similar results
        )
        return results
    except Exception as e:
        print(f"Error retrieving messages: {e}")
        return None

# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Create the persistent ChromaDB client
    client = create_chroma_client()
    if not client:
        exit()

    # 2. Get or create the 'chat_history' collection
    chat_collection = get_or_create_collection(client, "mrfrench-ai-coach")
    if not chat_collection:
        exit()

    # 3. Store some sample chat messages
    print("\n--- Storing chat messages ---")
    store_chat_message(chat_collection, "Hello, how is the weather today?", "chat1")
    store_chat_message(chat_collection, "I am a helpful assistant. How can I assist you?", "chat2")
    store_chat_message(chat_collection, "Do you have any information on current weather patterns?", "chat3")

    # 4. Search for similar messages
    print("\n--- Retrieving similar messages ---")
    query_text = "What is the weather like outside?"
    retrieved_messages = retrieve_chat_messages(chat_collection, query_text)

    if retrieved_messages:
        print(f"Query: '{query_text}'")
        print("Found similar messages:")
        for doc in retrieved_messages["documents"][0]:
            print(f"- {doc}")
            
    # You can inspect the 'chroma_db_data' folder created in your directory.
    # The data will remain there for future use.
