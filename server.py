from flask import Flask, render_template, url_for, request, jsonify
import pathlib
from pathlib import Path
import os
import json
import requests
from flask_cors import CORS
import torch
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BertTokenizer, BertModel
tokenizer_emb = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_emb = BertModel.from_pretrained("bert-base-multilingual-cased")
tokenizer_slm = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
model_slm = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True,)
pipe = pipeline("text-generation", model=model_slm, tokenizer=tokenizer_slm, max_new_tokens=256)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

model_kwargs = {'device': 'cpu'}
embeddings = HuggingFaceEmbeddings(model_kwargs=model_kwargs)
#tokenizer_lg = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
#tokenizer_lg = tokenizer_slm
#model_lg = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", device_map='auto', torch_dtype="auto", trust_remote_code=True,)
#model_lg=model_slm
###pipe_lg = pipeline("text-generation", model=model_lg, tokenizer=tokenizer_lg, max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Load the PDF file
#pdf_link = "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf"
pdf_link = "merged.pdf"
loader = PyPDFLoader(pdf_link, extract_images=False)
pages = loader.load_and_split()


# Split data into chunks
text_splitter = RecursiveCharacterTextSplitter(
   chunk_size = 4000,
   chunk_overlap  = 20,
   length_function = len,
   add_start_index = True,
)
chunks = text_splitter.split_documents(pages)
# Store data into database
db=Chroma.from_documents(chunks,embedding=embeddings,persist_directory="test_index")
db.persist()
 #Load the database
vectordb = Chroma(persist_directory="test_index", embedding_function = embeddings)
 #Load the retriver
retriever = vectordb.as_retriever(search_kwargs = {"k" : 3})
# Define the custom prompt template suitable for the Phi-3 model
qna_prompt_template="""<|system|>
You have been provided with the context and a question, try to find out the answer to the question only using the context information. If the answer to the question is not found within the context, return "I dont know" as the response.<|end|>
<|user|>
Context:
{context}

Question: {question}<|end|>
<|assistant|>"""
PROMPT = PromptTemplate(
   template=qna_prompt_template, input_variables=["context", "question"]
)

#Define the QNA chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
 # utility function for answer generation
def ask(question):
  prompt=""
  context = retriever.get_relevant_documents(question)
  answer = (chain({"input_documents": context, "question": question}, return_only_outputs=True))['output_text']
  return answer

#Setting functions for RAG
# Open the file in read mode
text=""""""
with open('context.txt', 'r') as file:
    # Read the entire content of the file
    text = file.read()

#model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = text.split('\n\n')
#embeddings = model.encode(sentences)
database=[]
vectorbase=[]
encoded_input = tokenizer_emb(sentences, return_tensors='pt', truncation=True, padding=True)

app = Flask(__name__)
CORS(app)

@app.route('/alive')
def beep():
    return jsonify("Hello, are you alive ?"), 200

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/process', methods=['POST'])
def process_message():
    data = request.get_json()
    user_message = data.get('message')

@app.route('/process_text', methods=['POST'])
def process_old():
    data = request.get_json()
    text = data.get('text', '')
    # Process the text (for example, convert to uppercase)
    processed_text = text.upper()
    return jsonify({'processed_text': processed_text})

@app.route('/send_conversation', methods=['POST'])
def process_text():
    prompt=""""""
    data = request.get_json()
    torch.cuda.empty_cache()
    for individual in data['conversation']:
        prompt=prompt+individual['sender']+": "+individual['text']+"\n\n"
        try:
          prompt=prompt+individual['bot']+": "+individual['text']+"\n\n"
        except:
          print("no bot")

    # Process the text (for example, convert to uppercase)
    processed_text = ask(prompt)
    # Split the text by the delimiter
    answer = (processed_text.split("<|assistant|>")[-1]).strip()

    torch.cuda.empty_cache()
    return jsonify({'processed_text': answer})

@app.route('/send_conversation_summarize', methods=['POST'])
def process_text_summarize():
    prompt=""""""
    data = request.get_json()
    torch.cuda.empty_cache()
    for individual in data['conversation']:
        prompt=prompt+individual['sender']+": "+individual['text']+"\n\n"
        try:
          prompt=prompt+individual['bot']+": "+individual['text']+"\n\n"
        except:
          print("no bot")

    prompt="Summarize Accordingly:\n\n"+prompt
    # Process the text (for example, convert to uppercase)
    processed_text = ask(prompt)
    # Split the text by the delimiter
    answer = (processed_text.split("<|assistant|>")[-1]).strip()

    torch.cuda.empty_cache()
    return jsonify({'processed_text': answer})

@app.route('/send_conversation_getinfo', methods=['POST'])
def process_text_getinfo():
    prompt=""""""
    data = request.get_json()
    torch.cuda.empty_cache()
    for individual in data['conversation']:
        prompt=prompt+individual['sender']+": "+individual['text']+"\n\n"
        try:
          prompt=prompt+individual['bot']+": "+individual['text']+"\n\n"
        except:
          print("no bot")

    query=prompt
    encoded_query = tokenizer_emb(query, return_tensors='pt', truncation=True, padding=True)
    similarities=[]
    for i in encoded_input['input_ids']:
      j=[]
      j.append(i)
      similarities.append(np.sum(np.dot(encoded_query['input_ids'].T, j)))
    print(np.asarray(similarities).shape)
    top_3_idx=np.argsort(similarities)
    top_3_idx
    prompt=""
    for i in top_3_idx:
     prompt=prompt+sentences[i]

    prompt="From the given context\n\n "+prompt+"\n\nAnswer this query \n\n"+query
    # Generate text 
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, num_beams=1, temperature=0.3, top_k=50, top_p=0.95, max_time=180)

    # Print the generated text
    answer = outputs[0]['generated_text'][len(prompt):].strip()
    torch.cuda.empty_cache()
    return jsonify({'processed_text': answer})

@app.route('/alive')
def beep():
    return jsonify("Hello, are you alive ?"), 200

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/summarize')
def summarize():
    return render_template('summarize.html')

@app.route('/getinfo')
def getinfo():
    return render_template('getinfo.html')

@app.route('/process', methods=['POST'])
def process_message():
    data = request.get_json()
    user_message = data.get('message')

@app.route('/send_conversation', methods=['POST'])
def process_text():
    prompt=""""""
    data = request.get_json()
    text = data.get('text', '')
    for individual in data['conversation']:
        prompt=prompt+individual['sender']+": "+individual['text']+"\n\n"
        try:
          prompt=prompt+individual['bot']+": "+individual['text']+"\n\n"
        except:
          print("no bot")
    # Process the text (for example, convert to uppercase)
    processed_text = """Microsoft’s SpeechT5 is a text-to-speech (TTS) model available on Hugging Face. This model is part of the SpeechT5 framework, which is designed for various spoken language processing tasks, including TTS1. The framework uses a unified encoder-decoder architecture to handle both speech and text inputs, making it versatile for tasks like speech synthesis, automatic speech recognition, and more1.

You can easily use the SpeechT5 model with the Hugging Face Transformers library. Here’s a quick """
    return jsonify({'processed_text': processed_text})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
