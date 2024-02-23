import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.llms import GooglePalm
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
import os
from dotenv import load_dotenv
import tempfile
from langchain.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.evaluation import load_evaluator
from pprint import pprint as print
from langchain.evaluation import EmbeddingDistance
from langchain.evaluation import Criteria
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from langchain.evaluation.comparison import LabeledPairwiseStringEvalChain
load_dotenv()

def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
   # history.append((query, result["answer"]))
    st.session_state['chat_history'].append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain,user_input):
    reply_container = st.container()
    container = st.container()

#with container:
    # with st.form(key='my_form', clear_on_submit=True):
    #user_input = st.chat_input("Question:",key='input')
        #submit_button = st.form_submit_button(label='Send')

    if user_input:
        with st.spinner('Generating response...'):
            output = conversation_chat(user_input, chain, st.session_state['chat_history'])
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")


def set_custom_prompt(select_model):
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt_template = """ You are a personal Bot assistant specialized in providing assistance related to Universal Robots oem products. When asked about your identity, respond by stating that you are a personal Bot assistant providing Universal Robots oem product-related assistance.

For questions unrelated to Universal Robots oem products, reply with 'I don't have knowledge about this. Please contact their customer service.'

Your task is to respond to user questions using information from the vector database without relying on your own knowledge. 
If an answer is found, always start with 'here are the steps' or directly address the context without using 'Based on the information provided in the document.' If an answer cannot be found in the vector database, simply state that you don't know and avoid making up an answer.
Additionally, if a user provides feedback on the steps given and indicates that they are not working, respond appropriately while maintaining the specified guidelines. If the query is linked to a previous question, consider the context and respond accordingly. Do not include links in your responses, and replace the term 'document' with 'knowledge.
Do not include links in your responses, and replace the term 'document' with 'knowledge.' Use bullet points only when necessary for creating lists.


Context: {context}
Question: {question}

"""
    prompt_template_llama = """ \<s>[INST]
    <<SYS>>You are a personal Bot assistant specialized in providing assistance related to Universal Robots oem products. When asked about your identity, respond by stating that you are a personal Bot assistant providing Universal Robots oem product-related assistance.

For questions unrelated to Universal Robots oem products, reply with 'I don't have knowledge about this. Please contact their customer service.'

Your task is to respond to user questions using information from the vector database without relying on your own knowledge. 
If an answer is found, always start with 'here are the steps' or directly address the context without using 'Based on the information provided in the document.' If an answer cannot be found in the vector database, simply state that you don't know and avoid making up an answer.
Additionally, if a user provides feedback on the steps given and indicates that they are not working, respond appropriately while maintaining the specified guidelines. If the query is linked to a previous question, consider the context and respond accordingly. Do not include links in your responses, and replace the term 'document' with 'knowledge.
Do not include links in your responses, and replace the term 'document' with 'knowledge.' Use bullet points only when necessary for creating lists.

Context: {context}
<<SYS>>Question: {question}[INST]
"""
    if select_model=='Llama 2 7b' or 'Llama 2 13b' or 'Llama 2 70b':
        prompt = PromptTemplate(
        template=prompt_template_llama, input_variables=["context", "question"]
    )
        return prompt

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt




def create_conversational_chain(vector_store,llm,select_model):
    load_dotenv()
    prompt = set_custom_prompt(select_model)
    # Create llm  
    #memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = ConversationBufferWindowMemory(k=2,memory_key = "chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory,combine_docs_chain_kwargs={"prompt": prompt})
    return chain


def evaluation_on_crit(query,reply,retriever,criteria):
  gptevaluator = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key="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")
  evaluator = LabeledCriteriaEvalChain.from_llm(
  llm=gptevaluator,
  criteria=criteria,
  )
  contexts=[]
  reference= contexts.append([docs.page_content for docs in retriever.get_relevant_documents(query)])

  eval_result = evaluator.evaluate_strings(
          prediction=reply,
          input=query,
          reference=contexts)
  return eval_result
def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs RAGs ChatBot :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)
    select_model=st.sidebar.selectbox("Choose a Model", ['Llama 2 7b', 'Llama 2 13b', 'Llama 2 70b','Palm','Open Ai'], key='select_model')
    if select_model=='Llama 2 7b':
        llm=Replicate(streaming = True,model = "meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4", 
    model_kwargs={"temperature": 0.02, "max_output_tokens": 700, "top_p": 1})
    elif select_model=='Llama 2 13b':
        llm=Replicate(streaming = True,model = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d", 
    model_kwargs={"temperature": 0.02, "max_output_tokens": 700, "top_p": 1})
    elif select_model=='Llama 2 70b':
        llm=Replicate(streaming = True,model = 'meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3', 
    model_kwargs={"temperature": 0.02, "max_output_tokens": 700, "top_p": 1})
    elif select_model=='Open Ai':
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key="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")
        #llm=OpenAI(
           # model_name="gpt-3.5-turbo",
           # temperature=0.02,
            ##max_tokens=700,
            #top_p=1)
    else:
        llm= GooglePalm(model="chat-bison", max_output_tokens=700, temperature=0.02,top_p=1)
        #llm=ChatVertexAI(model="chat-bison", max_output_tokens=1000, temperature=0.5,top_p=0.1)
    print(select_model)

    user_input = st.chat_input("Question:",key='input')
    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=40, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
        vector_store = Chroma.from_documents(text_chunks,embeddings,persist_directory="./db")


        # Create vector store
       # vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store,llm,select_model)
       # print("chain------------",chain)

    
            
        
        display_chat_history(chain,user_input)

if __name__ == "__main__":
    main()
