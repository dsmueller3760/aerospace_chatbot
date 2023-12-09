# ---
# USE IF IN DATABUTTON
# import databutton as db
# ---

import os
import queries
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
import openai
import time

from tqdm.auto import tqdm
from typing import Tuple

from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)

from canopy.tokenizer import Tokenizer
from canopy.knowledge_base import KnowledgeBase
from canopy.context_engine import ContextEngine
from canopy.chat_engine import ChatEngine
from canopy.llm.openai import OpenAILLM
# from canopy.llm.models import ModelParams
from canopy.models.data_models import Document, Messages, UserMessage, AssistantMessage
from canopy.models.api_models import ChatResponse

def chat(new_message: str, history: Messages) -> Tuple[str, Messages, ChatResponse]:
    messages = history + [UserMessage(content=new_message)]
    response = chat_engine.chat(messages)
    assistant_response = response.choices[0].message.content
    return assistant_response, messages + [AssistantMessage(content=assistant_response)], response

# ---
# USE IF IN DATABUTTON
# PINECONE_ENVIRONMENT=db.secrets.get('PINECONE_ENVIRONMENT')
# PINECONE_API_KEY=db.secrets.get('PINECONE_API_KEY')
# os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT
# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# ---

# Set secrets
PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

# Set the page title
st.set_page_config(
    page_title='Aerospace Chatbot: AMS w/Langchain',
)
st.title('Aerospace Mechanisms Chatbot')
with st.expander('''What's under the hood?'''):
    st.markdown('''
    This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
    * Source code: https://github.com/dsmueller3760/aerospace_chatbot/blob/main/scripts/setup_page_canopy.py
    * Uses pinecone canopy: https://www.pinecone.io/blog/canopy-rag-framework/
    * **Response time ~45 seconds per prompt**
    ''')

# Add a sidebar for input options
st.title('Input')
st.sidebar.title('Input Options')

# Add input fields in the sidebar
model_name=st.sidebar.selectbox('Model', ['gpt-3.5-turbo''gpt-3.5-turbo-16k','gpt-3.5-turbo','gpt-3.5-turbo-1106','gpt-4','gpt-4-32k'], index=1)
model_list={'gpt-3.5-turbo':4096,
            'gpt-3.5-turbo-16k':16385,
            'gpt-3.5-turbo-1106':16385, 
            'gpt-4':8192,
            'gpt-4-32k':32768}
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
n=None  # Not used. How many chat completion choices to generate for each input message.
top_p=None  # Not used. Only use this or temperature. Where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

k=st.sidebar.number_input('Number document chunks per query', min_value=1, step=1, value=15)
output_level=st.sidebar.selectbox('Level of Output', ['Concise', 'Detailed', 'No Limit'], index=2)
max_prompt_tokens=model_list[model_name]

# Vector databases
st.sidebar.title('Vector Database')
index_name=st.sidebar.selectbox('Index name', ['canopy--ams'], index=0)

# Embeddings
st.sidebar.title('Embeddings')
embedding_type=st.sidebar.selectbox('Embedding type', ['Openai'], index=0)
embedding_name=st.sidebar.selectbox('Embedding name', ['text-embedding-ada-002'], index=0)

# Add a section for secret keys
st.sidebar.title('Secret Keys')
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')

# ---
# USE IF IN DATABUTTON
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# ---

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    embeddings_model = OpenAIEmbeddings(model=embedding_name,openai_api_key=OPENAI_API_KEY)

    # Set up chat history
    qa_model_obj = st.session_state.get('qa_model_obj',[])
    message_id = st.session_state.get('message_id', 0)
    history = st.session_state.get('history',[])

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Process some items
    if output_level == 'Concise':
        out_token = 50
    else:
        out_token = 516

    # Display assistant response in chat message container
    if prompt := st.chat_input('Prompt here'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            message_placeholder = st.empty()

            with st.status('Generating response...') as status:
                t_start=time.time()
                message_id += 1
                st.write('Message: '+str(message_id))
                
                # Process some items
                if output_level == 'Concise':
                    max_generated_tokens = 50
                elif output_level == 'Detailed':
                    max_generated_tokens = 516
                else:
                    max_generated_tokens = None
                    
                # Inialize canopy
                Tokenizer.initialize()
                pinecone.init(
                    api_key=PINECONE_API_KEY,
                    environment=PINECONE_ENVIRONMENT
                )

                kb = KnowledgeBase(index_name=index_name,
                                   default_top_k=k)
                kb.connect()
                context_engine = ContextEngine(kb)
                llm=OpenAILLM(model_name=model_name)
                chat_engine = ChatEngine(context_engine,
                                        llm=llm,
                                        max_generated_tokens=max_generated_tokens,
                                        max_prompt_tokens=max_prompt_tokens)
                
                st.write('Searching vector database, generating prompt...')
                response, history, chat_response = chat(prompt, history)

                message_placeholder.markdown(response)
                t_delta=time.time() - t_start
                status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
        
        st.session_state['history'] = history
        st.session_state['qa_model_obj'] = qa_model_obj
        st.session_state['message_id'] = message_id
        st.session_state.messages.append({'role': 'assistant', 'content': response})

else:
    st.warning('No API key found. Add your API key in the sidebar under Secret Keys. Find it or create one here: https://platform.openai.com/api-keys')
    st.info('Your API-key is not stored in any form by this app. However, for transparency it is recommended to delete your API key once used.')