import streamlit as st
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from angle_emb import AnglE
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR


def get_pdf_text(pdf_docs):
  """Takes pdf documents and returns raw texts.

  Parameters:
    pdf_docs(list): List of pdf documents.

  Returns:
    text(string): Raw text formatted in one long string.

  """

  text = ""

  for pdf in pdf_docs:
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
      text += page.extract_text()

  return text


def get_text_chunks(raw_text, tokenizer):
  """ Takes raw text and returns text chunks.
  
  Parameters:
    raw_text(str): String of text from document.
    tokenizer(AutoTokenizer): Tokenizer from huggingface.

  
  Returns:
    chunks(list): List of chunks (str) of 505 tokens.
  
  """
  text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, 
    chunk_size=505, 
    chunk_overlap=50)
  
  chunks = text_splitter.split_text(raw_text)

  return chunks


def get_vector_store(text_chunks):
  """Takes text chunks and converts them into vectors and stores them.

  Parameters:
    text_chunks(dict[str]): Dictionary of strings containing chunks
  
  Returns:
    
  """
    #embeddings = OpenAIEmbeddings()
  embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

  vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

  return vector_store


def get_conversation_chain(vector_store):
  """Takes a vector store and
  
  """

  llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xxl")

  memory = ConversationBufferMemory(memory_key= 'chat_history', return_messages= True)

  conversation_chain = ConversationalRetrievalChain.from_llm(
      llm= llm,
      retriever = vector_store.as_retriever(),
      memory= memory

  )


  return conversation_chain


def handle_userinput(user_question):
  """########
  
  Parameters:
    user_question:
  """
  response = st.session_state.conversation({'question': user_question})
  st.session_state.chat_history = response['chat_history']

  return response


def main():
  load_dotenv()
  #API TOKEN GEBRUIKEN
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl", device_map = 'auto')


  st.set_page_config(page_title = "Power2X Query Tool",
                     page_icon = 'docs/Power2X-Logo.png')
  
  with st.columns(3)[1]:
    st.image('docs/Power2X-Logo.png', width= 200)


  if "conversation" not in st.session_state:
    st.session_state.conversation = None

    
  st.header("Use this tool to query PDF files")

  user_question = st.text_input("Ask a question about your document:")
  if user_question:
    response = handle_userinput(user_question)

    with st.expander('Response object'):
      st.write(response)

    with st.expander("Source text"):
      st.write(response.get_formatted_sources())
  #st.text_input("Ask a question here")
  

  with st.sidebar:
    st.subheader("Your documents")

    pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'", accept_multiple_files= True)

    if st.button("Process"):
      with st.spinner("Processing"):
        #GET TEXT
        raw_text = get_pdf_text(pdf_docs)

        #GET CHUNKS
        text_chunks = get_text_chunks(raw_text, tokenizer)

        # CREATE VECTOR STORES
        vector_store = get_vector_store(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vector_store)
        

if __name__ == '__main__':
  main()