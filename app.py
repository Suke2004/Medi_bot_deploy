from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from dotenv import load_dotenv
import os, logging, json
from datetime import datetime
from model import load_gemini

# LangChain & AI Imports
from src.helper import download_huggingface_model
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")

# Enable CORS
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_RESOURCES'] = {r"/*": {"origins": "*"}}
CORS(app)

# Constants
CHAT_LOG_FILE = "chat_logs.json"

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()

# Load embedding model and LLM
embeddings = download_huggingface_model()
llm = load_gemini()

# Prompt and chain setup
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

# Pinecone vector store and retrieval chain (DB-based)
db_docsearch = PineconeVectorStore.from_existing_index(index_name="medibotlarge", embedding=embeddings)
db_retriever = db_docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
db_rag_chain = create_retrieval_chain(db_retriever, question_answer_chain)

# ---------- UTILITIES ----------

def log_chat(question, answer):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer
    }
    if os.path.exists(CHAT_LOG_FILE):
        with open(CHAT_LOG_FILE, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(CHAT_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ---------- ROUTES ----------

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET"])
def chat():
    try:
        user_input = request.args.get("msg")

        if not user_input:
            return jsonify({"error": "Missing input"}), 400

        result = db_rag_chain.invoke({"input": user_input})
        answer = result.get("answer", "No answer generated.")
        log_chat(user_input, answer)
        return jsonify({"response": answer})

    except Exception as e:
        logging.exception("Chat error:")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/admin")
def admin_dashboard():
    try:
        with open(CHAT_LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []
    return render_template("admin.html", logs=logs)

# # ---------- MAIN ----------

# if __name__== '__main__':
#     app.run(host="0.0.0.0", port=8080, debug=True)
