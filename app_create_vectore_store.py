"""
Autor:    Ingmar Stapel
Datum:    20230913
Version:  0.1
Homepage: https://ai-box.eu/

This program shows how to create text embeddings and store them in a FAISS vector database.
I developed that program to get into the topic of creating text embeddings.
It is using LangChain, HuggingFace, and the FAISS vector data store for the text embeddings.

Everything runs locally NO API CALLS to OpenAI etc. ...

You may need two GPUs with 24 GB of video RAM or a single GPU with 48 GB of video RAM to have some fun.

Important:
Many thanks for the How-to guides from @alejandro_ao and @jamesbriggs on YouTube which I combined for that app.

@alejandro_ao
https://www.youtube.com/channel/UC1oXUA7qgs0GZc_yk46K2OQ

@jamesbriggs
https://www.youtube.com/channel/UCv83tO5cePwHMt1952IVVHw


"""

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
import os
import shutil
import re
import yaml

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text, in_separator, in_chunk_size, in_chunk_overlap, in_text_splitter_type):
    if in_text_splitter_type == "Character Text Splitter":
        # 20230904 old text splitter
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/character_text_splitter
        text_splitter = CharacterTextSplitter(
            separator="\n", # yeahhh just hard coded I know...
            chunk_size=int(in_chunk_size),
            chunk_overlap=int(in_chunk_overlap),
            length_function=len
        )
    else:
        # 20230904 new text splitter
        # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(in_chunk_size),
            chunk_overlap=int(in_chunk_overlap),
            length_function = len,
            #is_separator_regex = False, # did not fully understood that option yet.
        )

    chunks = text_splitter.split_text(text)
    return chunks

def set_vectorstore(text_chunks, embeddings, index_store):
  #check whether the text_chunks is documents, or text
  try:
    faiss_db = FAISS.from_documents(text_chunks, embeddings)  
  except Exception as e:
    faiss_db = FAISS.from_texts(text_chunks, embeddings)
  
  if os.path.exists(index_store):
    local_db = FAISS.load_local(index_store,embeddings)
    #merging the new embedding with the existing index store
    local_db.merge_from(faiss_db)
    local_db.save_local(index_store)
  else:
    faiss_db.save_local(folder_path=index_store)

def get_vectorstore_length(index_store, embeddings):
  test_index = FAISS.load_local(index_store, embeddings=embeddings)
  test_dict = test_index.docstore._dict
  return len(test_dict.values())  

def create_vectorstore(text_chunks, vectorstore_name_path, embedding_model_name, ):
    # Check this out https://huggingface.co/spaces/mteb/leaderboard
    embeddings = HuggingFaceInstructEmbeddings(model_name=embedding_model_name)
    set_vectorstore(text_chunks, embeddings, vectorstore_name_path)
    vectorstore = FAISS.load_local(vectorstore_name_path, embeddings)

    return vectorstore

def load_vectorstore(path, in_model_name):#, embedding_model_name):
    embedding_model_name = in_model_name # load_meta_embedding(path+'/', 'readme.txt')
    embeddings = HuggingFaceInstructEmbeddings(model_name=str(embedding_model_name))    
    st.session_state.placeholder = str(get_vectorstore_length(path, embeddings))
    vectorstore = FAISS.load_local(path, embeddings)
    return vectorstore

# Maybe I will change that coding to use the yaml library
def save_meta_embedding(path, file_name, embedding_model_name, in_chunk_size, in_chunk_overlap, 
    in_text_splitter_type, in_text_separator, in_vectorstore_length_function):
    try:
        with open(path+file_name, 'w') as f:
            f.write(str("Model name: " + embedding_model_name + "\nChunk size: " + in_chunk_size + 
            "\nChunk overlap: " + in_chunk_overlap + "\nText Splitter: " + in_text_splitter_type
            + "\nText seperator: " + in_text_separator + "\nlen function: " + in_vectorstore_length_function))
    except FileNotFoundError:
        print("The 'docs' directory does not exist")

def main():
    load_dotenv()
    st.set_page_config(page_title="Create vector store database",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore_length" not in st.session_state:
        st.session_state.vectorstore_length = None

    st.header("Create your local vector store database:")
    
    st.subheader("Load local vectore store DB")
    st.write("Load your local vectore store DB to add additional documents.")

    # Please check that static path first...
    vectorstore_path = st.text_input("Set your general vectorstore save path:", value="/home/ingmar/FAISS_vectore_store/")

    # Get a list of all subfolders in the main folder
    subfolders = [f.name for f in os.scandir(vectorstore_path) if f.is_dir()]

    # Convert the list of subfolders to a comma-separated string
    subfolders_string = ', '.join(subfolders)

    # I do not need to load the valuas via the yaml config. They will not change often...
    embedding_model = st.selectbox(
    'Select an embedding model:',
    ("BAAI/bge-large-en", "BAAI/bge-base-en", "hkunlp/instructor-xl")) 

    option = st.selectbox(
    'Select a vectorstore:',
    (list(subfolders)))    

    if st.button("Load vectorstore DB") and option:
        with st.spinner("Processing"):

            # now load the vectore store config
            load_path = vectorstore_path + option + "/config.yaml"
            try:
                with open(load_path) as f:
                    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
            except FileNotFoundError:
                print("The 'config.yaml' does not exist")

            # load local vector store
            vectorstore = load_vectorstore(vectorstore_path + option, config_yaml['Model name'])
            st.session_state.placeholder_chunk_size = str(config_yaml['Chunk size'])
            st.session_state.placeholder_chunk_overlap = str(config_yaml['Chunk overlap'])
            st.session_state.placeholder_vec_name = str(option)


    if st.button("Delete vectorstore DB", type="primary") and option:
        with st.spinner("Processing"):
            # load local vector store
            shutil.rmtree(vectorstore_path + option)

    st.session_state.vectorstore_length = st.text_input("Shows the length of the current vector store:", key="placeholder",)
    st.subheader("Add your documents")
    pdf_docs = st.file_uploader(
        "Upload your PDFs here and click on 'Process' to store them in the local vectore store.", accept_multiple_files=True)

    vectorstore_name = st.text_input("Your vectorstore name:", key="placeholder_vec_name",)    

    # I do not need to load the valuas via the yaml config. They will not change often...
    text_splitter_type = st.selectbox(
    'Select an text splitter type',
    ("Recursive Character Text Splitter", "Character Text Splitter")) 

    if text_splitter_type == "Character Text Splitter":
        text_separator = st.text_input("Your text separator:", value="\\n",)
    else:
        text_separator = "NoOK"
    vectorstore_chunk_size = st.text_input("Your vectorstore embedding chunk size (2000):", key="placeholder_chunk_size")
    vectorstore_chunk_overlap = st.text_input("Your vectorstore embedding chunk overlap (200):", key="placeholder_chunk_overlap")
    vectorstore_length_function = st.text_input("Your vectorstore embedding length function (len):", value="len",)
    if text_splitter_type == "Recursive Character Text Splitter":
        vectorstore_is_separator_regex = st.text_input("Set is separator regex (False / True):", value="False",)

    if st.button("Process") and vectorstore_name and vectorstore_chunk_size and vectorstore_chunk_overlap:
        vectorstore_name_path = vectorstore_path + vectorstore_name + "/"

        with st.spinner("Processing"):
            # get pdf text
            raw_text = get_pdf_text(pdf_docs)

            # get the text chunks
            text_chunks = get_text_chunks(raw_text, text_separator, vectorstore_chunk_size, vectorstore_chunk_overlap, text_splitter_type)

            # load / create vector store
            vectorstore = create_vectorstore(text_chunks, vectorstore_name_path, embedding_model)
            save_meta_embedding(vectorstore_name_path, 'config.yaml', embedding_model, vectorstore_chunk_size, 
            vectorstore_chunk_overlap, text_splitter_type, text_separator, vectorstore_length_function)

if __name__ == '__main__':
    main()
