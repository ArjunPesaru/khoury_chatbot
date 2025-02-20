import streamlit as st
import ollama
import faiss
import numpy as np
import pickle
import os

# Paths
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "data", "faiss_index")
VECTOR_DB_PATH = os.path.join(os.getcwd(), "data", "vector_db.pkl")

# Models
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest'

# Load FAISS index
INDEX = faiss.read_index(FAISS_INDEX_PATH)

# Load text chunks and embeddings
with open(VECTOR_DB_PATH, "rb") as f:
    VECTOR_DB = pickle.load(f)

def retrieve(query, top_n=3):
    """Retrieve the most relevant chunks for a query."""
    query_embedding_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    if "embedding" not in query_embedding_response:
        return []

    query_embedding = np.array(query_embedding_response["embedding"]).astype("float32").reshape(1, -1)
    _, indices = INDEX.search(query_embedding, top_n)
    
    return [VECTOR_DB[i][0] for i in indices[0] if i < len(VECTOR_DB)]

# Streamlit UI
st.title("📚 AI-Powered Chatbot")
st.markdown("Ask me anything! I will use stored knowledge to generate a response.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        retrieved_knowledge = retrieve(query)

        st.subheader("🔍 Retrieved Knowledge")
        for chunk in retrieved_knowledge:
            st.write(f"- {chunk}")

        instruction_prompt = f"""You are a highly knowledgeable and helpful chatbot.
Use all the available information to provide the most complete and detailed response possible relate to khoury colleg at boston. 
Feel free to combine different pieces of context to give a well-rounded answer.
If relevant, include examples, explanations, and additional insights from your knowledge.

Here is the retrieved information to help answer the user's question:
""" + "\n".join([f' - {chunk}' for chunk in retrieved_knowledge])

        # Generate response
        response_stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[{'role': 'system', 'content': instruction_prompt}, {'role': 'user', 'content': query}],
            stream=True,
        )

        # Collect response before displaying
        response_text = ""
        for chunk in response_stream:
            response_text += chunk["message"]["content"]

        # Display response
        st.subheader("🤖 Chatbot Response")
        st.write(response_text)
    else:
        st.warning("Please enter a question!")
