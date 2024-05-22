from flask import Flask, request, jsonify
import os
from langchain.document_loaders import PyPDFLoader
import re
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from waitress import serve
from llama_parse import LlamaParse
from langchain.schema import Document
from llama_index.core import SimpleDirectoryReader
import nest_asyncio

app = Flask(__name__)

data_path = 'data/proposals'
UPLOAD_FOLDER = 'data/proposals'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize lists to store data
sources = []
documents = []
page_contents = []

# Extract source and document information from file paths
def extract_file_data():
    global sources, documents, page_contents

    loaded_pdfs = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]

    load_dotenv()
    # set up parser
    parser = LlamaParse(
        result_type="markdown"  # "markdown" and "text" are available
    )
    
    file_extractor = {".pdf": parser}
    for pdf in loaded_pdfs:
        pages = SimpleDirectoryReader(input_files=[pdf], file_extractor=file_extractor).load_data(nest_asyncio.apply())
        pdf_name = re.search(r'[^/]+$', pdf).group(0)
        for document in pages:
            sources.append(pdf_name)
            metadata = {
                "source": f"{pdf_name}",
            }
            lp_parsed_documents = Document(page_content=document.text, metadata=metadata)
            documents.append(lp_parsed_documents)
            page_contents.append(document.text)

def initialize_app():
    extract_file_data()
    global vectordb
    df = pd.DataFrame({
        'Source': sources,
        'Document': documents,
        'Page Content': page_contents
    })

    token_text_splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    tokenized_docs = token_text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    persist_directory = './chroma/'
    vectordb = Chroma.from_documents(
        documents=tokenized_docs,
        embedding=embedding,
        persist_directory=persist_directory
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        chain_type='stuff'
    )

    return conversation_chain

conversation_chain = initialize_app()

@app.route('/test', methods=['GET'])
def test():
    return '<h1>API Working! :-)</h1>'

@app.route('/predict', methods=['POST'])
def predict():
    question = request.get_json().get('question')

    if question:
        if vectordb._client.list_collections():
            chat_response = conversation_chain({'question': question})
            results = {
                'question': f"{chat_response['question']}",
                'answer': f"{chat_response['answer']}",
                'chat_history': f"{chat_response['chat_history']}"
            }
            return jsonify(results)
        else:
            return {"error": "Please upload proposals"}
    else:
        return jsonify({'error': 'No question provided'}), 400

@app.route('/upload', methods=['POST'])
def upload_files():
    global conversation_chain

    files = request.files.getlist('file')

    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'One or more files have no selected file'}), 400
        else:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    conversation_chain = initialize_app()

    return jsonify({'message': 'Files uploaded successfully'}), 200

@app.route('/delete', methods=['DELETE'])
def delete_files():
    files_to_keep = ['sample.pdf']  
    upload_folder = app.config['UPLOAD_FOLDER']

    for filename in os.listdir(upload_folder):
        if filename not in files_to_keep:
            file_path = os.path.join(upload_folder, filename)
            os.remove(file_path)

    conversation_chain.memory.clear()
    vectordb.delete_collection()

    return jsonify({'message': 'Files deleted successfully'}), 200

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
