import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

load_dotenv()

def get_llm():
    """
    Возвращает экземпляр LLM.
    Приоритет: OpenAI (gpt-4o-mini) -> Groq (Llama-3).
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    # ПРИОРИТЕТ 1: OpenAI (GPT-4o-mini)
    if openai_key:
        return ChatOpenAI(
            model="gpt-4o-mini-2024-07-18",
            temperature=0.0,
            api_key=openai_key
        )
    
    # ПРИОРИТЕТ 2: Groq (Fallback)
    # Используется только если нет ключа OpenAI
    elif groq_key:
        print("Warning: Using Groq (Llama 3) as fallback.")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            api_key=groq_key
        )
    
    else:
        raise ValueError("CRITICAL ERROR: No API keys found in .env")

# Тест
if __name__ == "__main__":
    llm = get_llm()
    # Проверка, какая модель подключилась
    print(f"Connected to: {llm.model_name if hasattr(llm, 'model_name') else llm.model}")
    res = llm.invoke("Say 'Ready'")
    print(res.content)