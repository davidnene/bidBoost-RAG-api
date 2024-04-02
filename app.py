
from flask import Flask, request, jsonify
import os
from langchain.document_loaders import PyPDFLoader
import re
import pandas as pd
from langchain.text_splitter import TokenTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from waitress import serve



app = Flask(__name__)

data_path = 'data/proposals'

app = Flask(__name__)

UPLOAD_FOLDER = 'data/proposals'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#store file paths in a list
loaded_pdfs = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]


# Initialize lists to store data
sources = []
documents = []
page_contents = []

# Extract source and document information from file paths
if loaded_pdfs != []:
    for pdf in loaded_pdfs:
        loader = PyPDFLoader(pdf)
        pages = loader.load()
        pdf_name = re.search(r'[^/]+$', pdf).group(0)
    for document in pages:
        sources.append(pdf_name)
        documents.append(document)
        page_contents.append(document.page_content)
    df = pd.DataFrame({
    'Source': sources,
    'Document': documents,
    'Page Content': page_contents
    })

    # intialize token splitter
    token_text_splitter = TokenTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    # pass the splitter to the documents
    tokenized_docs = token_text_splitter.split_documents(documents)
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    persist_directory = './chroma/'

    # loading intializing chroma with both embeddings and tokens
    vectordb = Chroma.from_documents(
        documents=tokenized_docs,
        embedding=embedding,
        persist_directory=persist_directory
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
    llm = llm,
    retriever=vectordb.as_retriever(),
    memory=memory)





@app.route('/predict', methods=['POST'])
def predict():
        question = request.get_json()['question']
        print(question)
        chat_response = conversation_chain({'question': f'{question}'})
        results = {
            'question': f"{chat_response['question']}",
            'answer': f"{chat_response['answer']}",
            'chat_history': f"{chat_response['chat_history']}",

        }

        return jsonify(results)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')

    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    for file in files:
        if file.filename == '':
            return jsonify({'error': 'One or more files have no selected file'}), 400
        else:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'message': 'Files uploaded successfully'}), 200

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)

