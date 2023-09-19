import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.utils import embedding_functions
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader


def get_pdf_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

# combine step3 - embeddings and step4 - vector store @ chroma in this function
def get_vectorstore(text_chunks):
    # Step 3 - Embeddings
    model_name = "hkunlp/instructor-xl" # one of the best embedding algo (free), even better than OpenAI
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Step 4 - Vector Database, we use Chroma which is free
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0: # reminder 0 means even number which is human msg
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)
        else: # reminder 1 means odd number which is robot msg
            st.write(user_template.replace('{{MSG}}', message.content), unsafe_allow_html=True)

def get_conversation_chain(vector_store):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain


def main():
    load_dotenv()

    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = None

    st.set_page_config(page_title='Chat with your PDFs',
                       page_icon=':books')
    
    st.header('YP-AI05 PDF-GPT\n This app allows you to chat with your PDF :books:')
    user_question = st.text_input('Ask a question about your pdf')
    if user_question:
        handle_userinput(user_question)
    st.write(css, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader('Your PDF:')
        pdf_docs = st.file_uploader(
            'Upload your PDF here and click on "Process"', accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                # convert pdf into chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store (after embedding process)
                vectorstore = get_vectorstore(text_chunks)

                # combine vectorstore with chat history + llm
                st.session_state.conversation = get_conversation_chain(vectorstore)





if __name__ == '__main__':
    main()