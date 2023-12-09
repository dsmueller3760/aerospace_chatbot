import streamlit as st
import os

# Set up page
st.set_page_config(
    page_title="Aerospace Chatbot: AMS",
)
st.title("Aerospace Chatbot Homepage")
st.markdown("Code base: https://github.com/dsmueller3760/aerospace_chatbot")
st.markdown('---')
st.title("Chatbots")
st.markdown("""
Chatbots for aerospace mechanisms symposia, using all available papers published since 2000
* Aerospace Mechanisms Chatbot, Langchain: Uses langchain QA retrieval https://databutton.com/v/71z0llw3/Aerospace_Mechanisms_Chat_Bot_Langchain
* Aerospace Mechanisms Chatbot, Canopy: Uses pinecone's canopy tool https://databutton.com/v/71z0llw3/Aerospace_Mechanisms_Chat_Bot_Canopy
""")
st.subheader("AMS")
'''
This chatbot will look up from all Aerospace Mechanism Symposia in the following location: https://github.com/dsmueller3760/aerospace_chatbot/tree/main/data/AMS
* Available models: https://platform.openai.com/docs/models
* Model parameters: https://platform.openai.com/docs/api-reference/chat/create
* Pinecone: https://docs.pinecone.io/docs/projects#api-keys
* OpenAI API: https://platform.openai.com/api-keys
'''

# # Establish secrets
# PINECONE_ENVIRONMENT=os.getenv('PINECONE_ENVIRONMENT')
# PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')