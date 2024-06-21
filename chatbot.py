import streamlit as st   # backend or frontend framework, to create apps easily
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter #  splitting text only thats why RCTS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
OPENAI_API_KEY='sk-proj-U79eiw7A2yNtx0llKjnkT3BlbkFJWcAUYLn8sWVu3uwIm'

#  Upload PDF files
st.header("My first Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")

#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)   # pdf reader just reading file not showing
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() # whatever written in file, comes into text variable now (we can see file now)
        # st.write(text)  # we can see our text


#Break it into chunks for better training and computation
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,  # length of each chunk is 1000 characters
        chunk_overlap=150, # 150 characters of previous chunk will be into next chunk for relevancy
        length_function=len # lenght of object is a function
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    
# generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings) #intializing FAISS(Facebook AI Semantic Search) $ storing chunks and embeddings

# # get user question
    user_question = st.text_input("Type Your question here")

#     # do similarity search
#     if user_question:
#         match = vector_store.similarity_search(user_question) #user's qns get embedded and do semantic search with all chunks and return similar chunks
#         #st.write(match)

#         #define the LLM
#         llm = ChatOpenAI(
#             openai_api_key = OPENAI_API_KEY,
#             temperature = 0,   # o means no randon, relevant to context....higher no means it is generating random things
#             max_tokens = 1000,  # limit of response...for  1000 , it would return response of around 750 words
#             model_name = "gpt-3.5-turbo"
#         )

#         #output results
#         #chain -> take the question, get relevant document, pass it to the LLM, generate the output
#         chain = load_qa_chain(llm, chain_type="stuff") #LLM is used here just like we use different ML models
#         response = chain.run(input_documents = match, question = user_question)
#         st.write(response)
