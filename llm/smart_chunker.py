from langchain_community.chat_models import ChatOpenAI
import openai
from langchain import LLMChain
from langchain import PromptTemplate

class Chunk:
    def __init__(self, content):
        self.page_content = content

text_splitter_prompt = """
You are a text splitter designed to split text into chunks. Unlike a mechanical text splitter that splits chunks based on token count, you will split them based on the following rules:
1. Keep logical ideas together. Let's say that you are given the following chunk of text to split:
    'Jenny was thirsty. She had not re-filled her water bottle in six hours. She was lost in the desert and knew she would die soon'.
    You should split this into three chunks:
    A. 'Jenny was thirsty.'
    B. 'Jenny had not re-filled her water bottle in six hours.' (Note here that we replaced 'she' with 'Jenny'. This keeps the idea self-contained - we could retrieve this chunk and know who 'she' refers to)
    C. 'Jenny was lost in the desert and knew that she would die soon because she was running out of water.' (note here that we add context to the sentence, even though it's not in the origional text. In retrieval augmented generation algorithms, the context returned does not need to be the exact text, but it needs to provide precise context. This ensures that the context (Jenny running out of water) is kept with the thought (she thought she would die soon).).
2. Keep sentences together. If a sentence is split across two chunks, it will be difficult to understand the meaning of the sentence. For example, if you are given the following chunk of text to split:
3. Throw out garbage data. The data you see has been scraped from a webpage, and hyperlinks did not transfer. You should remove any text that is not part of the main content. For example, if you are given the following chunk of text to split:
    'Jenny was thirsty. She had not re-filled her water bottle in six hours. She was lost in the desert and knew she would die soon. Click here to learn more about the desert.'
    You should split this into the three chunks from above, but you should remove the last sentence, as it's not part of the main content.
4. Handle code examples with care: If the text includes code examples or technical content, keep the entire example or related technical details together in one chunk. This ensures the coherence and usability of the code when retrieved.
5. Preserve context with technical content: When splitting technical content, include necessary context or comments within the same chunk to make the code or technical details comprehensible without additional external information.
6. Adjust for readability and retrieval: Ensure that each chunk is readable on its own and provides value for retrieval purposes. This might involve adding context or restructuring sentences to make them standalone.

Output each chunk and use %% as your delimiter. This data is being fed straight to the database, so do not say ANYTHING before or after the message. If you see text with chunks a, b and c your response should be a%%b%%c

Remember, you should be responding with the chunked text, not the example. Chunk the text below:

"""

def split_chunks_responsibly(doc_object, target_chunk_size):
    # print out the first 300 characters of the doc we're chunking
    text = doc_object.page_content
    print(f"Beginning to chunk document. First 300 characters: {text[:300]}")
    model = "gpt-4-1106-preview"
    # model = "gpt-3.5-turbo-0125"
    llm = ChatOpenAI(model=model, openai_api_key=openai.api_key)
    prompt_template = PromptTemplate(
        input_variables=["text_splitter_prompt", "text", "target_chunk_size"],  # Define the input variables your prompt uses
        template=text_splitter_prompt + """
            %Here is the data to chunk:%
            {text}
            
        Keep the chunks of text as close to the same length as possible. You should use {target_chunk_size} tokens as your ceiling for how large chunks can be, but it is more important to keep ideas together than it is to keep the chunks the same length. If you have to split a chunk into two, do so, but try to keep the chunks as close to the same length as possible. 
        """
    )    
    print("Created prompt template")
    chain = LLMChain(llm=llm, prompt=prompt_template)
    
    # Run the chain with the appropriate variables
    print("Getting llm response")
    llm_response = chain.run(text_splitter_prompt=text_splitter_prompt, text=text, target_chunk_size=target_chunk_size)
    print("Got response!")
    broken_out_chunks = llm_response.split("%%")
    
    print("Broken out chunk length ", len(broken_out_chunks))
    return broken_out_chunks