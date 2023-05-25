#from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,GPTSimpleVectorIndex, PromptHelper
from llama_index import LLMPredictor, ServiceContext
import sys
#from google.colab import drive
import os
import pyttsx3
import streamlit as st

# Set the page title
st.set_page_config(page_title="MedEzy")

st.sidebar.title('MedEzy')
if st.sidebar.button("New Chat"):
    clear_generated()

API_KEY=str(st.sidebar.text_input("Your OpenAI API key :",placeholder="Enter to submit"))
os.environ["OPENAI_API_KEY"] = API_KEY



def open_google_maps():
    # Open Google Maps with current location for doctors nearby
    webbrowser.open('https://www.google.com/maps/search/doctors+nearby/')



def text2speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    # Convert text to speech
    engine.say(text)
    engine.runAndWait()
    
   

def construct_index(directory_path):
  # set maximum input size
  max_input_size = 4096
  # set number of output tokens
  num_outputs = 256
  # set maximum chunk overlap
  max_chunk_overlap = 20
  # set chunk size limit
  chunk_size_limit = 600

  prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

  # define LLM
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=num_outputs , prompt="You are a medical bot,only answer questions related to medical field for other questions which are not related to medical say i don't know i am medical bot"))
  
  documents = SimpleDirectoryReader(directory_path).load_data()
  
  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
  index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
  
  index.save_to_disk('index.json')
  
  return index

def ask_bot(query):
  input_index = 'index.json'
  index = GPTSimpleVectorIndex.load_from_disk(input_index)
  response = index.query(query)
  return response

def clear_generated():
    st.session_state["generated"] = []
    st.session_state['past'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Set the page header
st.title("MedEzy")
# Create static text input box
query= st.text_input("🤖Ask AI", placeholder="Enter to submit")

output=" "
if API_KEY:
    st.sidebar.success('API key entered successfully.')
    st.sidebar.info(" Medical knowledge is constantly evolving, so please keep in mind that my responses are based on the information available up until September 2021.")
    index = construct_index("D:\DSA\codehelp\contents")
    if query:
        output = ask_bot(query)
        # store the output
        st.session_state.past.append(query)
        st.session_state.generated.append(str(output))
    

    with st.expander("Conversation", expanded=True):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.info(st.session_state["past"][i],icon="🧐")
            st.success(st.session_state["generated"][i], icon="🤖")

    text2speech(str(output))
else:
    st.sidebar.error("Enter your OpenAI API key!!")
    st.sidebar.info(" Medical knowledge is constantly evolving, so please keep in mind that my responses are based on the information available up until September 2021.")    
    st.success("""👋 Welcome to the MedEzy Bot! 🩺

I'm here to provide you with accurate and reliable information on various medical topics. Whether you have questions about symptoms, treatments, medications, preventive care, or nutrition, I'm here to help. Please remember to consult a healthcare professional for personalized advice.

Ask me anything related to medical queries, and I'll do my best to assist you! """,icon="🤖")
    


