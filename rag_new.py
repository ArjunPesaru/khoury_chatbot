import ollama
import os
import faiss
import numpy as np
import pickle

# Paths
DATASET_PATH = os.path.join(os.getcwd(), "data", "corpus.txt")
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "data", "faiss_index")
VECTOR_DB_PATH = os.path.join(os.getcwd(), "data", "vector_db.pkl")

# Load dataset
with open(DATASET_PATH, 'r', encoding='utf-8') as file:
    dataset = file.readlines()
    print(f'✅ Loaded {len(dataset)} entries from {DATASET_PATH}')

# Models
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest'

# FAISS vector index
DIMENSIONS = 768  # Adjust based on model output dimensions
INDEX = faiss.IndexFlatL2(DIMENSIONS)  # L2 (Euclidean) distance index
VECTOR_DB = []  # Store (text, embedding)

def add_chunk_to_database(chunk):
    """Embeds a text chunk and adds it to FAISS index."""
    try:
        if not chunk.strip():
            return

        truncated_chunk = chunk[:1000]  # Avoid large text chunks

        embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=truncated_chunk)
        if "embedding" in embedding_response:
            embedding = np.array(embedding_response["embedding"]).astype("float32").reshape(1, -1)
            VECTOR_DB.append((truncated_chunk, embedding))
            INDEX.add(embedding)
            print(f"✅ Added chunk: {chunk[:50]}... (truncated)")
        else:
            print(f"⚠️ Embedding failed for chunk: {chunk[:50]}...")
    
    except Exception as e:
        print(f"❌ Error embedding chunk: {chunk[:50]}... - {str(e)}")

# Embed all dataset chunks
for chunk in dataset:
    add_chunk_to_database(chunk)

# Save FAISS index and vector database
faiss.write_index(INDEX, FAISS_INDEX_PATH)
with open(VECTOR_DB_PATH, "wb") as f:
    pickle.dump(VECTOR_DB, f)

print(f"✅ Model saved! FAISS index stored at: {FAISS_INDEX_PATH}")

# CLI Chatbot Loop
def retrieve(query, top_n=3):
    """Retrieve the most relevant chunks for a query."""
    query_embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    if "embedding" not in query_embedding_response:
        return []

    query_embedding = np.array(query_embedding_response["embedding"]).astype("float32").reshape(1, -1)
    _, indices = INDEX.search(query_embedding, top_n)
    
    return [VECTOR_DB[i][0] for i in indices[0] if i < len(VECTOR_DB)]

while True:
    input_query = input("\nAsk me a question (or type 'exit' to quit): ")
    if input_query.lower() == "exit":
        print("👋 Exiting chatbot. Have a great day!")
        break

    retrieved_knowledge = retrieve(input_query)

    print("\n🔍 Retrieved Knowledge:")
    for chunk in retrieved_knowledge:
        print(f" - {chunk}")

    instruction_prompt = f"""You are a highly knowledgeable chatbot.
Use all the available information to provide a complete response.

Here is the retrieved information:
""" + "\n".join([f' - {chunk}' for chunk in retrieved_knowledge])

    # Generate response
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # Collect full response before printing
    response_text = ""
    for chunk in stream:
        response_text += chunk["message"]["content"]

    # Print full response
    print("\n🤖 Chatbot Response:")
    print(response_text)
