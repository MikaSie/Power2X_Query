import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def get_pdf_text(pdf_docs: list) -> str:
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


def get_text_chunks(raw_text):
  text_splitter = CharacterTextSplitter(

  )




def main():

  st.set_page_config(page_title = "Power2X Query Tool",
                     page_icon = 'docs/Power2X-Logo.png')
  
  with st.columns(3)[1]:
    st.image('docs/Power2X-Logo.png', width= 200)

  st.header("Use this tool to query PDF files")
  st.text_input("Ask a question here")

  with st.sidebar:
    st.subheader("Your documents")

    pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'")

    if st.button("Process"):
      with st.spinner("Processing"):
        #GET TEXT
        raw_text = get_pdf_text(pdf_docs)

        #GET CHUNKS

        # CREATE VECTOR STORES




if __name__ == '__main__':
  main()