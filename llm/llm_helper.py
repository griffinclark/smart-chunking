import json
from langchain import LLMChain, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from astra.cassandra_helper import astraSession, create_vector_db_with_cassandra, ASTRA_DB_KEYSPACE
import openai  # Importing the OpenAI library, used for accessing GPT models
# Loading the prompt template from the file (assuming it's stored as a string in prompt_template variable)


llama_index_prompt_template = """
You are a helpful, friendly Llama Index team member helping me understand how to use Llama Index's toolkit. I will ask you questions, and you'll respond with answers based on source information passed to your further down in the prompt, with a bias towards using Llama Index tools. Your answers should be consice, friendly, and include all relevant information that I need.

Be specific in your approach - don't list alternate methods. Pick one approach for solving each problem.

Make sure to output every step of the process, including the final answer, so I can see how you got there. Your output should follow the format:
```High level summary
Step 1: [Description of step 1]
Step 2: [Description of step 2]
etc. 

ONLY use modules from Llama Index that you see in the context below. If there's a Llama Index model that you want to use but you don't see it in the context, don't use it

Do not explicitly mention my level of technical expertise, role, relationship to Llama Index, etc.

"""

response_text =""

def fetch_documents_from_cassandra( session, keyspace, limit):
    print("Using passed in keyspace ", keyspace)
    # Using cosine similarity search, get the data we need
    select_stmt = f"SELECT text FROM {keyspace}.text_embeddings0 LIMIT %s;"
    rows = session.execute(select_stmt, (limit,))
    # The relevant documents from our vector database
    documents = [row.text for row in rows]
    return documents

def get_response_from_query(query, k, user_role, enable_smart_chunking, problem="Llama Index"):
    # Read user roles from the JSON file
    with open('llm/user_roles.json') as file:
        user_roles = json.load(file)
    
    # Fetch documents from Cassandra
    keyspace = ASTRA_DB_KEYSPACE if not enable_smart_chunking else "smart_chunking"

    print("Using keyspace ", keyspace)
    docs = fetch_documents_from_cassandra(astraSession, keyspace, k)
    
    if not openai.api_key:
        raise Exception("OPENAI_API_KEY environment variable not set.")
    
    # Join documents and prepare the prompt, ensuring no encoding to bytes here
    docs_page_content = " ".join([doc.replace("\n", " ") for doc in docs])
    
    # Define the LLM and the prompt
    llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai.api_key, temperature=0)
    
    # Building the complete prompt
    complete_prompt = f"""
    {llama_index_prompt_template}
    
    Role Description: {user_roles.get(user_role, "").replace("{", "{{").replace("}", "}}")}
    
    Question: {query.replace("{", "{{").replace("}", "}}")}
    Context: {docs_page_content.replace("{", "{{").replace("}", "}}")}
    """
    
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template=complete_prompt
    )
    
    # Create the prompt chain and run
    chain = LLMChain(llm=llm, prompt=prompt)
    llm_response = chain.run(question=query, docs=docs_page_content)
    response_text = "\n".join(llm_response)  # Append the response to response_text
    return llm_response, docs
