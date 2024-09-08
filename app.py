from src.helper import DEFAULT_SYSTEM_PROMPT, instructions
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import  PyPDFLoader 
from langchain_core.messages import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from flask import Flask,request,render_template, jsonify

from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")
from src.run_local import initialize_llm
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
embeddings = HuggingFaceInferenceAPIEmbeddings(api_key= HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
#Load the PDF File
def load_file(file_path):
    loader= PyPDFLoader(file_path)
    documents=loader.load()
    return documents

# Splitting the file and store it into vector DB
def chunking_vectordb(documents):
    #Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    #Load the Embedding Model
    #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    vector_store=FAISS.from_documents(text_chunks, embeddings)
    return vector_store



template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow up Input: {question}
Standalone questions: """

CONDENSE_QUESTION_PROMPT = PromptTemplate(template=template, input_variables=["question"])

def generate(path):
    llm = initialize_llm()
    documents = load_file(path)
    vector_store=chunking_vectordb(documents)
    qa = ConversationalRetrievalChain.from_llm(llm= llm,retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                               condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True, 
                                               verbose=False)
    
    return qa


application=Flask(__name__)
app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('first.html')

@app.route('/start', methods=['GET','POST'])
def start():
    if request.method == 'POST':
        os.makedirs("data", exist_ok= True)
        file = request.files['file']
        print(file)
        if file:
            file_path = os.path.join("data/" + secure_filename(file.filename))
            file.save(file_path)
            documents = load_file(file_path)
            vector_store=chunking_vectordb(documents)
            vector_store.save_local("faiss")
            return render_template("index.html")
        

@app.route('/get_answer', methods=["GET", "POST"])
def get_answer():

    if request.method == 'POST':
        user_input = request.form['question']
        llm = initialize_llm()
        store=FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)
        chain = ConversationalRetrievalChain.from_llm(llm= llm,retriever=store.as_retriever(search_kwargs={'k': 2}),
                                               condense_question_prompt=CONDENSE_QUESTION_PROMPT,return_source_documents=True, 
                                               verbose=False)
        chat_history = []
        result=chain.invoke({"question":user_input,"chat_history":chat_history})
        chat_history.extend(
        [
        HumanMessage(content= user_input),
        AIMessage(content=result["answer"])
        ])
        print(f"Answer:{result}")
        
    return render_template("index.html", results= str(result['answer']))
    # return jsonify({"response": str(result['answer']) })
            


if __name__ == "__main__":
     app.run(debug= True, host='0.0.0.0', port=5000)

    # chat_history = []
    # response = generate("data\RAGPaper.pdf")
    # query = "What is RAG-Sequence Model??"
    # print(response.invoke({"question":query,"chat_history":chat_history}))


