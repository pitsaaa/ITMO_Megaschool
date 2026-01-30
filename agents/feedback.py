from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm import get_llm
from utils.state import InterviewState

def feedback_node(state: InterviewState):
    print("--- Feedback Generation ---")
    
    llm = get_llm()
    messages = state['messages']
    
    # Собираем контекст
    candidate_info = state.get('candidate_info', {})
    name = candidate_info.get('name', 'Кандидат')
    stack = candidate_info.get('stack', 'General')
    level = candidate_info.get('level', 'Junior')
    
    # История диалога для анализа
    # Превращаем сообщения в текст: "Interviewer: ... \n Candidate: ..."
    conversation_text = ""
    for msg in messages:
        role = "Candidate" if msg.type == 'human' else "Interviewer"
        conversation_text += f"{role}: {msg.content}\n"

    system_prompt = f"""
    Ты — Технический Лид, проводящий финальную оценку интервью.
    
    КАНДИДАТ: {name}
    ЗАЯВЛЕННЫЙ УРОВЕНЬ: {level}
    СТЕК: {stack}
    
    ТВОЯ ЗАДАЧА:
    Проанализируй лог интервью и напиши ПОДРОБНЫЙ отчет в формате MARKDOWN.
    Это должно быть единое поле текста.
    
    СТРУКТУРА ОТЧЕТА (Используй Markdown заголовки ## и списки):
    
    1. ## Технический ревью
       - Оцени сильные и слабые стороны кандидата, проявленные в ответах.
       - Какие темы он знает хорошо? Где плавает? (SQL, Python, Архитектура и т.д.)
       
    2. ## Итоговый Грейд
       - Соответствует ли он уровню {level}?
       - Какой грейд ты бы дал (Junior/Middle/Senior)?
       
    3. ## План развития (Roadmap)
       - Список конкретных тем и технологий, которые нужно подтянуть.
       - Рекомендации (книги, курсы, пет-проекты).
       
    4. ## Заключение
       - Итоговое слово: нанимаем или нет?
       - Вежливое прощание.

    ВАЖНО:
    - Не используй JSON.
    - Пиши сразу красивый, отформатированный текст.
    - Язык: Русский.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Вот лог интервью:\n{conversation_text}\n\nСоставь отчет.")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        feedback_markdown = chain.invoke({})
    except Exception as e:
        print(f"❌ Feedback Error: {e}")
        feedback_markdown = "## Ошибка генерации отчета\nК сожалению, не удалось сформировать фидбэк."

    return {
        "final_feedback": feedback_markdown
    }