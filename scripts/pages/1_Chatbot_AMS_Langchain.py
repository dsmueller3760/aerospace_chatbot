# ---
# USE IF IN DATABUTTON
# import databutton as db
# ---

import os
import queries
import pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
import openai
import time

from dotenv import load_dotenv,find_dotenv,dotenv_values
load_dotenv(find_dotenv(),override=True)

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
    * Source code: https://github.com/dsmueller3760/aerospace_chatbot/blob/main/scripts/setup_page_langchain.py
    * Uses custom langchain functions with QA retrieval: https://js.langchain.com/docs/modules/chains/popular/chat_vector_db_legacy
    * All prompts will query entire database unless 'filter response with last received sources' is activated.
    * **Repsonse time ~10 seconds per prompt**.
    ''')
filter_toggle=st.checkbox('Filter response with last received sources?')

# Add a sidebar for input options
st.title('Input')

# Add input fields in the sidebar
st.sidebar.title('Input options')
output_level = st.sidebar.selectbox('Level of Output', ['Concise', 'Detailed'], index=1)
k = st.sidebar.number_input('Number of items per prompt', min_value=1, step=1, value=4)
search_type = st.sidebar.selectbox('Search Type', ['similarity', 'mmr'], index=1)
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=2.0, value=0.0, step=0.1)
verbose = st.sidebar.checkbox('Verbose output')
chain_type = st.sidebar.selectbox('Chain Type', ['stuff', 'map_reduce'], index=0)

# Vector databases
st.sidebar.title('Vector database')
index_type=st.sidebar.selectbox('Index type', ['Pinecone'], index=0)
index_name=st.sidebar.selectbox('Index name', ['canopy--ams'], index=0)

# Embeddings
st.sidebar.title('Embeddings')
embedding_type=st.sidebar.selectbox('Embedding type', ['Openai'], index=0)
embedding_name=st.sidebar.selectbox('Embedding name', ['text-embedding-ada-002'], index=0)

# Add a section for secret keys
st.sidebar.title('Secret keys')
OPENAI_API_KEY = st.sidebar.text_input('OpenAI API Key', type='password')

# Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
    embeddings_model = OpenAIEmbeddings(model=embedding_name,openai_api_key=OPENAI_API_KEY)

    # Set up chat history
    qa_model_obj = st.session_state.get('qa_model_obj',[])
    message_id = st.session_state.get('message_id', 0)

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

    # Define LLM parameters and qa model object
    llm = OpenAI(temperature=temperature,
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=out_token)
    qa_model_obj=queries.QA_Model(index_name,
                    embeddings_model,
                    llm,
                    k,
                    search_type,
                    verbose,
                    filter_arg=False)

    # Display assistant response in chat message container
    if prompt := st.chat_input('Prompt here'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)
        with st.chat_message('assistant'):
            message_placeholder = st.empty()

            with st.status('Generating response...') as status:
                t_start=time.time()

                # Process some items
                if output_level == 'Concise':
                    out_token = 50
                else:
                    out_token = 516

                # Define LLM parameters and qa model object
                llm = OpenAI(temperature=temperature,
                                openai_api_key=OPENAI_API_KEY,
                                max_tokens=out_token)

                message_id += 1
                st.write('Message: '+str(message_id))
                
                if message_id>1:
                    qa_model_obj=st.session_state['qa_model_obj']
                    qa_model_obj.update_model(llm,
                                        k=k,
                                        search_type=search_type,
                                        verbose=verbose,
                                        filter_arg=filter_toggle)
                    if filter_toggle:
                        filter_list = list(set(item['source'] for item in qa_model_obj.sources[-1]))
                        filter_items=[]
                        for item in filter_list:
                            filter_item={'source': item}
                            filter_items.append(filter_item)
                        filter={'$or':filter_items}
                
                st.write('Searching vector database, generating prompt...')
                qa_model_obj.query_docs(prompt)
                ai_response=qa_model_obj.result['answer']
                message_placeholder.markdown(ai_response)
                t_delta=time.time() - t_start
                status.update(label='Prompt generated in '+"{:10.3f}".format(t_delta)+' seconds', state='complete', expanded=False)
        
        st.session_state['qa_model_obj'] = qa_model_obj
        st.session_state['message_id'] = message_id
        st.session_state.messages.append({'role': 'assistant', 'content': ai_response})

else:
    st.warning('No API key found. Add your API key in the sidebar under Secret Keys. Find it or create one here: https://platform.openai.com/api-keys')
    st.info('Your API-key is not stored in any form by this app. However, for transparency it is recommended to delete your API key once used.')