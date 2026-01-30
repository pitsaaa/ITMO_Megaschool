import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser # Используем строковый парсер
from pydantic import BaseModel, Field
from typing import Literal

from utils.llm import get_llm
from utils.state import InterviewState

def observer_node(state: InterviewState):
    print("--- Observer Working ---")
    
    messages = state['messages']
    
    # --- ЛОГИКА ХОЛОДНОГО СТАРТА ---
    # Если в истории только 1 сообщение (команда старта от main.py) или 0
    # То анализировать нечего. Возвращаем заглушку.
    if len(messages) <= 1:
        print("Observer: Первый ход. Пропуск анализа.")
        return {
            "observer_analysis": {
                "thoughts": "Начало интервью. Ожидаю первый вопрос.",
                "is_hallucination": False,
                "consistency_violation": False,
                "is_deep_dive": False,
                "is_role_reversal": False,
                "intent_to_leave": False,
                "answer_quality": "medium" # Нейтрально
            },
            "current_turn_thoughts": ["[Observer]: (Start of Interview)"]
        }

    last_user_text = messages[-1].content
    last_bot_msg = messages[-2].content if len(messages) > 1 else "Начало интервью"
    
    # Защита от пустого ввода
    if not last_user_text.strip():
        print("Observer: Пустой ввод.")
        return {
            "observer_analysis": {},
            "current_turn_thoughts": ["[Observer]: Пустой ввод."]
        }
    
    llm = get_llm()
    messages = state['messages']
    
    # 1. ЗАЩИТА ОТ ПУСТОГО НАЧАЛА
    if not messages:
        return {"current_turn_thoughts": ["[Observer]: Нет сообщений для анализа."]}

    last_user_text = messages[-1].content
    last_bot_msg = messages[-2].content if len(messages) > 1 else "Начало интервью"
    
    # 2. ЗАЩИТА ОТ ПУСТОГО ВВОДА ПОЛЬЗОВАТЕЛЯ
    if not last_user_text.strip():
        print("⚠️ Observer: Пустой ввод пользователя.")
        return {
            "observer_analysis": {},
            "current_turn_thoughts": ["[Observer]: Пустой ввод."]
        }
    
    candidate_info = state.get('candidate_info', {})
    stack = candidate_info.get('stack', 'General')
    level = candidate_info.get('level', 'Junior')
    
    # --- ПРОМПТ ---
    # Убрали Pydantic, пишем структуру JSON прямо в промпте текстом (это надежнее для StrParser)
    system_prompt = """
    Ты — Строгий Поведенческий Аналитик (Observer).
    
    КОНТЕКСТ:
    - Кандидат: {name} ({level} {role})
    - Стек: {stack}
    
    ТВОЯ ЗАДАЧА: Проанализировать последний ответ кандидата.
    
    АЛГОРИТМ ПРОВЕРКИ (ФЛАГИ):
    1. is_hallucination: True, если выдумал факты/библиотеки.
    2. consistency_violation: True, если противоречит себе или грейду.
    3. is_deep_dive: True, если уходит в дебри не по теме.
    4. is_role_reversal: True, если задает встречные вопросы/перехватывает инициативу.
    5. intent_to_leave: True, если пишет "Стоп", "Хватит", "Закончим" или по-другому открыто проявляет желание остановить интервью.
    
    ФОРМАТ ВЫВОДА (ТОЛЬКО JSON, БЕЗ ЛИШНЕГО ТЕКСТА):
    {{
        "thoughts": "Твой краткий анализ на русском (макс 4 предл).",
        "is_hallucination": false,
        "consistency_violation": false,
        "is_deep_dive": false,
        "is_role_reversal": false,
        "intent_to_leave": false,
        "answer_quality": "medium" 
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Контекст (вопрос бота): {last_bot_msg}\nОтвет кандидата: {last_user_text}\n\nJSON:")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        # 3. ВЫЗОВ МОДЕЛИ
        raw_response = chain.invoke({
            "name": candidate_info.get('name', 'Candidate'),
            "level": level,
            "role": candidate_info.get('role', 'Developer'),
            "stack": stack,
            "last_bot_msg": last_bot_msg,
            "last_user_text": last_user_text
        })
        
        # ДЕБАГ: Смотрим в консоль, что пришло
        print(f"🔧 Observer Raw Output: {raw_response[:100]}...") 

        # 4. РУЧНАЯ ЧИСТКА JSON
        cleaned_json = raw_response.replace("```json", "").replace("```", "").strip()
        analysis_result = json.loads(cleaned_json)

    except Exception as e:
        print(f"❌ Observer JSON Error: {e}")
        # Fallback, чтобы не молчал
        analysis_result = {
            "thoughts": f"Ошибка анализа (JSON Error). Текст ответа был слишком сложным.",
            "is_hallucination": False,
            "consistency_violation": False,
            "is_deep_dive": False,
            "is_role_reversal": False,
            "intent_to_leave": False,
            "answer_quality": "medium"
        }

    # --- СБОРКА ЛОГА ---
    flags = []
    if analysis_result.get('is_hallucination'): flags.append("HALLUCINATION")
    if analysis_result.get('consistency_violation'): flags.append("CONTRADICTION")
    if analysis_result.get('is_deep_dive'): flags.append("OFF-TOPIC")
    if analysis_result.get('is_role_reversal'): flags.append("ROLE_REVERSAL")
    if analysis_result.get('intent_to_leave'): flags.append("STOP_REQUEST")
    
    flag_str = f" [FLAGS: {', '.join(flags)}]" if flags else ""
    
    # Используем .get() на случай, если поле thoughts называется по-другому
    thought_text = f"[Observer]: {analysis_result.get('thoughts', 'Analysis done')}{flag_str}"

    return {
        "observer_analysis": analysis_result,
        "current_turn_thoughts": [thought_text] # Создаем чистый список мыслей для этого хода
    }