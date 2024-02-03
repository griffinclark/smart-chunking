import json
from langchain import LLMChain, PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from astra.cassandra_helper import astraSession, create_vector_db_with_cassandra, ASTRA_DB_KEYSPACE
import openai  # Importing the OpenAI library, used for accessing GPT models
# Loading the prompt template from the file (assuming it's stored as a string in prompt_template variable)


prompt_template = """
You are a helpful, friendly Llama Index team member helping me understand how to use Llama Index's toolkit. I will ask you questions, and you'll respond with answers based on source information passed to your further down in the prompt. Your answers should be consice, friendly, and include all relevant information that I need.

If at any point you find yourself mentioning a Llama Index product or tool, give me a sentence or two of information about what the tool is and why it's cool/unique/helpful before getting into the rest of your answer

Do not explicitly mention my level of technical expertise, role, relationship to Llama Index, etc.

If the context provided in the prompt is not sufficient to answer the question, let the user know that you did not get the required context and ask them to tweak their query. Then, say "DEBUG INFO" and provide the context you received.
"""

response_text =""

def fetch_documents_from_cassandra( session, keyspace, limit):
    print("Using passed in keyspace ", keyspace)
    # Using cosine similarity search, get the data we need
    select_stmt = f"SELECT text FROM {keyspace}.text_embeddings2 LIMIT %s;"
    rows = session.execute(select_stmt, (limit,))
    # The relevant documents from our vector database
    documents = [row.text for row in rows]
    return documents

#what is going on here?
def get_response_from_query(query, k, user_role, enable_smart_chunking, problem):
    # Read user roles from the JSON file
    with open('llm/user_roles.json') as file:
        user_roles = json.load(file)
    
    # Fetch documents from Cassandra
    keyspace = ""
    if problem == "Llama Index":
        keyspace = ASTRA_DB_KEYSPACE if not enable_smart_chunking else "smart_chunking"
    elif problem == "Uber Q/K": 
        keyspace = "uber_qk" if not enable_smart_chunking else "uber_qk_smart_chunking"
    else:
        print("Invalid problem type")
    print("Using keyspace ", keyspace)
    docs = fetch_documents_from_cassandra(astraSession, keyspace, k)
    if not openai.api_key:
        raise Exception("OPENAI_API_KEY environment variable not set.")
    
    # Join documents and prepare the prompt
    docs_page_content = " ".join([doc.replace("\n", " ") for doc in docs]).encode("utf-8")
    
    # Define the LLM and the prompt
    llm = ChatOpenAI(model="gpt-4-1106-preview", openai_api_key=openai.api_key)

    # Building the complete prompt
    complete_prompt = f"""
    {prompt_template}
    
    Role Description: {user_roles.get(user_role, "")}

    Question: {query}
    Context: {docs_page_content}
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
