# Import my secure bundle
import json
import os
from dotenv import load_dotenv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pathlib
import uuid
from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cassandra.cluster import Session
import re

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Load our .env file so that we have access to the keys we need
load_dotenv()

# Embeddings are what we use to make our vector database readable by the LLM (GPT-4). See LangChain documentation for more details
embeddings = OpenAIEmbeddings()


try:
    # Get the directory in which the current script is located
    current_script_path = pathlib.Path(__file__).parent.absolute()

    secure_connect_bundle_path = current_script_path / 'secure-connect-smart-chunking.zip'

    cloud_config = {
        'secure_connect_bundle': str(secure_connect_bundle_path)
    }
except:
    print("No secure bundle found. Exiting...")
    exit(1)

# Get the directory in which the current script is located
current_script_path = pathlib.Path(__file__).parent.absolute()

# Construct the absolute path of the astra-token.json file
astra_token_file_path = current_script_path / 'astra-token.json'

with open(str(astra_token_file_path)) as f:
    secrets = json.load(f)

# Variables that we'll use for connecting to our VDB
CLIENT_ID = secrets["clientId"]
CLIENT_SECRET = secrets["secret"]
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
astraSession = cluster.connect()

session_id = uuid.uuid4()
mem_key="chat_history"

## If we want to add in message history (to create a chatbot), uncomment and integrate this code. However, it's not needed for the demo
# message_history = CassandraChatMessageHistory(
#     session_id=session_id,
#     session=astraSession,
#     keyspace=ASTRA_DB_KEYSPACE,
# )
# cass_buff_memory = ConversationBufferMemory(
#    memory_key=mem_key,
#    chat_memory=message_history
# )

def create_table_if_not_exists(session, keyspace):
    # Create the table if it doesn't exist
    create_stmt = f"""
    CREATE TABLE IF NOT EXISTS {keyspace}.text_embeddings (
        id UUID PRIMARY KEY,
        text TEXT,
        embedding LIST<FLOAT>  -- Adjust the data type for 'embedding' as needed
    );
    """
    try:
        session.execute(create_stmt)
        print("Table 'text_embeddings' created or already exists in keyspace", keyspace)
    except Exception as e:
        print("Error creating table:", e)


def sanitize_text(text):
    # Remove unsafe characters for Cassandra
    sanitized_text = re.sub(r"[^\w\s]", "", text)

    # Remove unsafe characters for Python
    sanitized_text = sanitized_text.replace("'", "")

    return sanitized_text

def create_vector_db_with_cassandra(folder_path: str, astraSession: Session):
    # When we update the data, we want to clear the existing data in the table and rebuild it from scratch (because I'm lazy and didn't want to write the code to update the data)
    create_table_if_not_exists(astraSession, ASTRA_DB_KEYSPACE)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)

    # Truncate the table to clear existing data
    truncate_stmt = f"TRUNCATE {ASTRA_DB_KEYSPACE}.text_embeddings;"

    # Execute the truncate statement
    astraSession.execute(truncate_stmt)
    print("Cleared existing data in the table.")

    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return
    # For every text file in our folder, add it into the vector database
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Process only text files
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                transcript = file.read()

                # Sanitize the transcript
                sanitized_transcript = sanitize_text(transcript)

                # Wrap the sanitized transcript in a Document object
                doc_object = Document(sanitized_transcript)
                docs = text_splitter.split_documents([doc_object])

                for _, doc in enumerate(docs):
                    # Embed the content of the document
                    embedding = embeddings.embed_documents([doc.page_content])[0]  # First item of the result

                    # Generate a unique ID for the document
                    doc_id = uuid.uuid4()

                    # Store the document text and its embedding in Cassandra
                    insert_stmt = astraSession.prepare(f"""
                        INSERT INTO {ASTRA_DB_KEYSPACE}.text_embeddings (id, text, embedding)
                        VALUES (?, ?, ?)
                    """)
                    astraSession.execute(insert_stmt, [doc_id, doc.page_content, embedding])

    return ASTRA_DB_KEYSPACE
