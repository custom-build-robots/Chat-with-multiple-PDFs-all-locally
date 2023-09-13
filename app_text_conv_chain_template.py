"""
Autor:    Ingmar Stapel
Datum:    20230913
Version:  0.1
Homepage: https://ai-box.eu/

This program shows how to run a large language model locally.
I developed that program to get into the topic of Chat with PDF files / chat with data.
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
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from torch import cuda, bfloat16
import transformers
import os
import yaml

fass_db_path="/home/ingmar/FAISS_vectore_store/"

def load_meta_embedding(path, file_name):
    path = path + file_name
    try:
        with open(path, 'r') as f:
            return f.readline().strip()
    except FileNotFoundError:
        print("The 'docs' directory does not exist")

def load_vectorstore(path, in_model_name):
    embedding_model_name = in_model_name #load_meta_embedding(path+'/', 'readme.txt')
    embeddings = HuggingFaceInstructEmbeddings(model_name=str(embedding_model_name))    
    vectorstore = FAISS.load_local(path, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, model_in, quantization_in):
    model_id = model_in
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    load_dotenv()
    # initializing HF
    hf_auth = os.getenv("hf_auth")

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    if quantization_in == "int4":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            #quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )

    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
 #       use_fast=True,
        use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=4096, #512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        # https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa
        retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50, 'score_threshold': 0.8}),        
        memory=memory,
        verbose=True
    )

    return conversation_chain


def get_sentiment_chain(model_in, quantization_in):
    model_id = model_in

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    load_dotenv()
    # initializing HF
    hf_auth = os.getenv("hf_auth")

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    if quantization_in == "int4":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            #quantization_config=bnb_config,
            device_map='auto',
            use_auth_token=hf_auth
        )

    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
 #       use_fast=True,
        use_auth_token=hf_auth
    )

    generate_text = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=4096, #512,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True)

    prompt_template = "Perform sentiment analysis on the following text: {question}  Determine whether it is positive, negative, or neutral."

    conversation_sentiment_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template),
        #chain_type="stuff",
        # https://python.langchain.com/docs/use_cases/question_answering/how_to/vector_db_qa
        #retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50, 'score_threshold': 0.8}),        
        memory=memory,
        verbose=True
    )

    return conversation_sentiment_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})

    if st.session_state.conv_debug:
        st.write(response)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    try:
        with open("config.yaml") as f:
            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
    except FileNotFoundError:
        print("The 'config.yaml' does not exist")


    st.set_page_config(page_title="Work with your data",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "conv_debug" not in st.session_state:
        st.session_state.conv_debug = None
    if "option_prompt_template" not in st.session_state:
        st.session_state.option_prompt_template = None


    st.header("Work with a large language model :sunglasses:")
    user_question = st.text_area("Your input goes here (questions, coding request, text for analysis...):", key="user_question_text_area")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Load local vector store")
        st.write("Load your local vector store to chat with already processed PDF files.")

        # Get a list of all subfolders in the main folder
        subfolders = [f.name for f in os.scandir(fass_db_path) if f.is_dir()]

        # Convert the list of subfolders to a comma-separated string
        subfolders_string = ', '.join(subfolders)

        option_model = st.selectbox(
        'Select a LLM model:',
        (config_yaml['llm_list']))  

        option_quantization = st.selectbox(
        'Select a quantization option:',
        ('int4', 'No thanks - I will not call the admin'))  

        option_prompt_template = st.selectbox(
        'Select a prompt template:',
        ('Chat with a vector store', 'Help me to code', 'Text sentiment analysis'))  

        if option_prompt_template != "Text sentiment analysis":
            option_vec_store = st.selectbox(
            'Select a vector store:',
            (list(subfolders)))    

        if st.button("Initialize app"):
            with st.spinner("Processing"):
                if option_prompt_template == "Text sentiment analysis":
                    # create conversation chain
                    st.session_state.conversation = get_sentiment_chain(option_model, option_quantization)
                
                elif option_vec_store:
                    # now load the vectore store config
                    load_path = fass_db_path + option_vec_store + "/config.yaml"
                    try:
                        with open(load_path) as f:
                            config_yaml = yaml.load(f, Loader=yaml.FullLoader)
                    except FileNotFoundError:
                        print("The 'config.yaml' does not exist")

                    # load local vector store
                    vectorstore = load_vectorstore(fass_db_path + option_vec_store, config_yaml['Model name'])
                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, option_model, option_quantization)    

if __name__ == '__main__':
    main()
