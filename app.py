import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    chunk_overlap=50,
)
  
  chunks = text_splitter.split_text(raw_text)

  return chunks



def main():
  load_dotenv()
  #API TOKEN GEBRUIKEN
  tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xxl")


  st.set_page_config(page_title = "Power2X Query Tool",
                     page_icon = 'docs/Power2X-Logo.png')
  
  with st.columns(3)[1]:
    st.image('docs/Power2X-Logo.png', width= 200)

  st.header("Use this tool to query PDF files")
  st.text_input("Ask a question here")

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




if __name__ == '__main__':
  main()