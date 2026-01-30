import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm import get_llm
from utils.state import InterviewState

def expert_node(state: InterviewState):
    print("--- Expert Working ---") 
    
    llm = get_llm()
    messages = state['messages']
    
    # --- ИЗВЛЕЧЕНИЕ ДАННЫХ ---
    candidate_info = state.get('candidate_info', {})
    stack = candidate_info.get('stack', 'General')
    level = candidate_info.get('level', 'Junior')
    
    covered_topics = state.get('topics_covered', [])
    observer_analysis = state.get('observer_analysis', {})
    current_thoughts = state.get('current_turn_thoughts', [])
    
    # Контекст
    last_user_msg = messages[-1].content
    last_bot_msg = messages[-2].content if len(messages) > 1 else "Intro"

    # --- ПОДГОТОВКА JSON OBSERVER ---
    try:
        observer_json_str = json.dumps(observer_analysis, ensure_ascii=False, indent=2)
    except:
        observer_json_str = "Анализ недоступен"

    # --- ПРОМПТ ---
    system_prompt = """
    Ты — Технический Лид (Expert). 
    
    ТВОЯ ЦЕЛЬ: Сформировать JSON с планом действий.
    
    КОНТЕКСТ:
    - Стек: {stack} ({level})
    - Прошлый вопрос бота: "{last_bot_msg}"
    - Обсужденные темы: {covered_topics}
    
    ОТЧЕТ НАБЛЮДАТЕЛЯ:
    {observer_report}

    ИНСТРУКЦИЯ:
    1. ПРОВЕРКА НА СТАРТ: Если прошлый вопрос бота похож на "Intro", "Начало" или "Intro Message":
       - НЕ ОЦЕНИВАЙ ОТВЕТ (так как кандидат только подтвердил готовность).
       - Игнорируй флаги наблюдателя на этом шаге.
       - Твоя задача: Сразу задать первый вводный вопрос по заявленному стеку.

    2. ЕСЛИ ЭТО НЕ СТАРТ (ОБЫЧНЫЙ ХОД):
       - Если есть флаг Hallucination -> Инструкция: "Опровергни факт и спроси источник."
       - Если есть флаг Stop -> Инструкция: "Заверши интервью." (Topic: Conclusion)
       - Если ответ ХОРОШИЙ -> ВЫБЕРИ НОВУЮ ТЕМУ. Не спрашивай одно и то же!
       - Если ответ СЛАБЫЙ -> Задай уточняющий вопрос.

    ФОРМАТ ВЫВОДА (ТОЛЬКО ЧИСТЫЙ JSON, БЕЗ MARKDOWN):
    {{
        "thoughts": "Твои мысли (макс 3 предл).",
        "instruction": "Что именно спросить у кандидата (прямая речь для интервьюера).",
        "topic_name": "Название темы (например: 'SQL Joins' или 'Global Lock').",
        "difficulty_adjustment": "same"
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Ответ кандидата: {last_user_msg}")
    ])

    # Используем StrOutputParser (получаем просто строку), а не JsonOutputParser
    chain = prompt | llm | StrOutputParser()

    try:
        # Вызываем модель
        raw_response = chain.invoke({
            "level": level,
            "stack": stack,
            "covered_topics": ", ".join(covered_topics),
            "observer_report": observer_json_str,
            "last_bot_msg": last_bot_msg,
            "last_user_msg": last_user_msg
        })
        
        # ДЕБАГ: Видим, что ответила модель на самом деле
        print(f"🔧 Expert Raw Output: {raw_response[:100]}...") 

        # --- РУЧНАЯ ЧИСТКА JSON ---
        # Удаляем ```json и ``` если они есть
        cleaned_json = raw_response.replace("```json", "").replace("```", "").strip()
        
        expert_plan = json.loads(cleaned_json)

    except Exception as e:
        print(f"❌ Expert JSON Error: {e}")
        # Если всё сломалось, явно говорим интервьюеру сменить тему
        expert_plan = {
            "thoughts": "Ошибка парсинга. Меняю тему принудительно.",
            "instruction": "Ответ принят. Давайте перейдем к следующей теме. Расскажите, что вы знаете про базы данных?",
            "topic_name": "Emergency Topic",
            "difficulty_adjustment": "same"
        }

    # --- ОБНОВЛЕНИЕ STATE ---
    
    expert_thought_str = f"[Expert]: {expert_plan.get('thoughts', '...')} [Strat: {expert_plan.get('difficulty_adjustment', 'same')}]"
    updated_thoughts_list = current_thoughts + [expert_thought_str]
    
    topic_name = expert_plan.get('topic_name', 'General')
    new_topics = [] 
    
    if topic_name not in ["Current Topic", "Conclusion", "General", "Emergency Topic"] and topic_name not in covered_topics:
        new_topics.append(topic_name)
        
    should_finish = False
    if observer_analysis.get('intent_to_leave', False) or topic_name == "Conclusion":
        should_finish = True

    return {
        "expert_plan": expert_plan,
        "topics_covered": new_topics,
        "current_turn_thoughts": updated_thoughts_list,
        "finished": should_finish
    }