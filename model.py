from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_deepseek import ChatDeepSeek
# from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv


def load_gemini():
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.4,
        max_output_tokens=512,
        api_key=GEMINI_API_KEY,
    )
    return llm

# def load_deepseek():
#     load_dotenv()
    
#     deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
#     if not deepseek_api_key:
#         raise EnvironmentError("Missing DEEPSEEK_API_KEY in .env file.")
    
#     # Optional: Set it in the environment if some internal library depends on it
#     os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key

#     llm = ChatDeepSeek(
#         model="deepseek-chat-3.5",
#         temperature=0.4,
#         max_tokens=512,  # Corrected key name from `max_output_tokens` to `max_tokens` if needed
#         api_key=deepseek_api_key,
#     )
#     return llm

# def load_model():
#     load_dotenv()
    
#     api_key = os.getenv("API_KEY")
#     api_base = os.getenv("API_BASE")  # Endpoint URL like https://api.deepseek.com/v1
#     model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")  # Default fallback model

#     if not api_key or not api_base:
#         raise EnvironmentError("Missing API_KEY or API_BASE in .env file.")
    
#     # Optional: Set for libraries that look for these in env vars
#     os.environ["OPENAI_API_KEY"] = api_key
#     os.environ["OPENAI_API_BASE"] = api_base

#     llm = ChatOpenAI(
#         openai_api_key=api_key,
#         openai_api_base="https://api.openai.com/v1",
#         model_name="gpt-4-o-mini",
#         temperature=0.4,
#         max_tokens=512,
#     )
    
#     return llm