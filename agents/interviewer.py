from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm import get_llm
from utils.state import InterviewState

def interviewer_node(state: InterviewState):
    print("--- Interviewer Working ---") 
    
    llm = get_llm()
    messages = state['messages']
    
    # Защита от сбоев Эксперта
    expert_plan = state.get('expert_plan') or {}
    instruction = expert_plan.get('instruction', "Поблагодари и задай следующий вопрос.")
    
    # --- ДИНАМИЧЕСКИЙ КОНТЕКСТ ---
    # Если в истории уже есть сообщения (кроме стартового), значит диалог идет.
    # Обычно: [System, Human(Start), AI(Intro), Human(Answer)...]
    # Если длина > 2, значит мы уже познакомились.
    is_ongoing_conversation = len(messages) > 2
    
    if is_ongoing_conversation:
        greeting_rule = "СТРОГИЙ ЗАПРЕТ НА ПРИВЕТСТВИЯ: Не здоровайся. Сразу к вопросу."
    else:
        greeting_rule = "ЭТО НАЧАЛО: Обязательно поздоровайся и представься."

    system_prompt = """
    Ты — Технический рекрутер Алиса.
    
    ТВОЯ ЗАДАЧА:
    Перефразировать инструкцию Эксперта в живой диалог.
    
    ВХОДНАЯ ИНСТРУКЦИЯ:
    "{instruction}"
    
    ПРАВИЛА ОФОРМЛЕНИЯ:
    1. {greeting_rule}
    2. ВАЖНО: Твоя цель — ПОЛУЧИТЬ ОТВЕТ. Каждое твоё сообщение (кроме прощания) ДОЛЖНО заканчиваться вопросительным предложением.
    3. Не читай лекции и не объясняй термины сама, если тебя об этом не просили. Твоя роль — спрашивать.
    4. Будь лаконична (2-3 предложения).
    5. Тон: Профессиональный, но дружелюбный.
    6. Если инструкция говорит завершить — попрощайся.
    7. Язык: Русский
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Сгенерируй ответ.") 
    ])
    
    # Передаем переменные
    chain = prompt | llm | StrOutputParser()
    
    try:
        response_text = chain.invoke({
            "instruction": instruction,
            "greeting_rule": greeting_rule
        })
    except Exception as e:
        print(f"❌ Interviewer Error: {e}")
        response_text = "Отлично. Давайте двигаться дальше."

    # --- ОБНОВЛЕННАЯ СБОРКА ЛОГА ---
    last_user_message = messages[-1].content if messages else ""
    is_start = "Начни интервью" in last_user_message or "Intro" in last_user_message

    if is_start:
        # Это самый первый запуск. Алиса говорит "Привет! Вопрос 1". 
        # Мы ничего не пишем в лог, но запоминаем Вопрос 1 в last_bot_msg.
        return {
            "messages": [AIMessage(content=response_text)],
            "last_bot_msg": response_text
        }

    # Если это НЕ старт, значит пользователь ответил на какой-то вопрос.
    # Этот вопрос лежит в state['last_bot_msg'].
    
    current_logs = state.get('internal_log', [])
    turn_id = len(current_logs) + 1
    
    thoughts_list = state.get('current_turn_thoughts', [])
    combined_thoughts = "\n".join(thoughts_list)
    
    # Берем старый вопрос из состояния для записи в текущий ход
    asked_question = state.get('last_bot_msg', "Вводный вопрос")

    log_entry = {
        "turn_id": turn_id,
        "agent_visible_message": asked_question, # Теперь тут ПРАВИЛЬНЫЙ вопрос (на который ответили)
        "user_message": last_user_message,
        "internal_thoughts": combined_thoughts
    }
    
    return {
        "messages": [AIMessage(content=response_text)],
        "internal_log": [log_entry],
        "last_bot_msg": response_text # Сохраняем НОВЫЙ вопрос для следующего хода
    }