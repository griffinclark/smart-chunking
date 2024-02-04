import streamlit as st
from astra.cassandra_helper import astraSession, create_vector_db_with_cassandra
from dotenv import load_dotenv
import os
import json  # Import json for reading the user_roles.json file
from llm.llm_helper import get_response_from_query, response_text
import openai
import time
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Other imports remain the same

def token_count(text):
    return len(encoding.encode(text))
load_dotenv()

session = astraSession
openai.api_key = os.getenv("OPENAI_API_KEY")

# Read user roles from the JSON file
with open('llm/user_roles.json') as file:
    user_roles = json.load(file)

# Create the UI
st.set_page_config(page_title="Llama Index Helper", layout="wide")

st.title("Smart Chunking Demo: Llama Index Helper")

with st.sidebar:
    st.header("Llama Index Helper Options")

    # User role selector
    user_role = st.selectbox(
        "Select your role",
        options=['Choose one...'] + list(user_roles.keys()),
        index=0  # Default to 'Choose one...'
    )

    # Only show other options if a valid user role is selected
    if user_role != 'Choose one...':
        enable_smart_chunking = st.checkbox("Enable Smart Chunking")

        user_input = st.text_area(
            "Enter your query here",
            height=300
        )

        k = st.number_input(
            "k",
            help="Number of documents to fetch from the database",
            min_value=1,
            max_value=300,
            value=60,
            step=1
        )

        if st.button("Generate Response"):
            start_time = time.time()
            llm_response, docs = get_response_from_query(user_input, k, user_role, enable_smart_chunking)
            response_text = response_text.join(llm_response)
            total_tokens = sum(token_count(doc) for doc in docs)
    
    # End timer
            end_time = time.time()
            generation_time = end_time - start_time
            st.write(f"Generation Time: {generation_time:.2f} seconds")
            st.write(f"Total Context Token Count: {total_tokens}")

        if st.button("Rebuild Database"):
            keyspace = create_vector_db_with_cassandra(
                folder_path="./data",
                astraSession=astraSession,
                enable_smart_chunking=enable_smart_chunking
            )   
            st.text("Database rebuilt!")

# Display the generated response in the main page
if response_text:
    st.subheader("Response")
    st.write(response_text)