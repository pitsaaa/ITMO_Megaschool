import streamlit as st
import uuid
from langchain_core.messages import HumanMessage

from main import build_graph, save_logs


# --- CONFIG HELPER ---
def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# --- НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(page_title="AI Interview Coach", page_icon="🤖")

st.title("AI Tech Interviewer")
st.caption("Разработано на LangGraph")


# --- CSS ---
st.markdown("""
<style>
    .stExpander {
        background-color: #fff9c4 !important;
        border: 1px solid #ffe082 !important;
        border-radius: 8px !important;
        color: #333333 !important;
    }
    .stExpander summary {
        color: #333333 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE INIT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_state" not in st.session_state:
    st.session_state.graph_state = None

if "app" not in st.session_state:
    st.session_state.app = build_graph()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "interview_active" not in st.session_state:
    st.session_state.interview_active = False


# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Настройки")

    name = st.text_input("ФИО", "Петров Сережа")
    role = st.text_input("Позиция", "C++ Developer")
    level = st.selectbox("Грейд", ["Junior", "Middle", "Senior"])
    stack = st.text_input("Стек", "C++, PostgreSQL")
    log_file = st.text_input("Имя файла лога", "interview_log.json")

    start_btn = st.button("Начать интервью", type="primary")

    if start_btn:
        # Полный сброс UI-сессии
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interview_active = True

        initial_input = {
            "messages": [HumanMessage(content="Начни интервью.")],
            "candidate_info": {
                "name": name,
                "role": role,
                "level": level,
                "stack": stack
            },
            "topics_covered": [],
            "internal_log": [],
            "finished": False,
            "last_bot_msg": None
        }

        result = st.session_state.app.invoke(
            initial_input,
            config=get_config()
        )

        st.session_state.graph_state = result

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["messages"][-1].content,
            "thoughts": "Инициализация интервью…"
        })

        st.rerun()


# --- RENDER CHAT ---
for msg in st.session_state.messages:

    if msg["role"] == "assistant":
        with st.chat_message("assistant", avatar="👩‍💼"):
            if msg.get("thoughts"):
                with st.expander("🧠 Мысли Observer / Expert"):
                    st.markdown(f"_{msg['thoughts']}_")
            st.write(msg["content"])

    elif msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(msg["content"])

    elif msg["role"] == "system":
        st.markdown(
            f"""
            <div style="
                background-color: #f0f2f6;
                color: #1f2937;               /* ← ВАЖНО */
                border-left: 5px solid #22c55e;
                padding: 14px 16px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
                margin-top: 14px;
            ">
            {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )


# --- INPUT HANDLING ---
if st.session_state.interview_active:
    input_text = st.chat_input("Ваш ответ…")
else:
    st.chat_input("Интервью завершено", disabled=True)
    input_text = None


if input_text:

    # 1. Рисуем сообщение пользователя
    st.session_state.messages.append({
        "role": "user",
        "content": input_text
    })

    # 2. Запуск графа
    with st.spinner("Алиса думает…"):
        result = st.session_state.app.invoke(
            {"messages": [HumanMessage(content=input_text)]},
            config=get_config()
        )

    st.session_state.graph_state = result

    # 3. Ответ интервьюера
    last_bot_msg = result["messages"][-1].content

    current_thoughts = ""
    if result.get("internal_log"):
        current_thoughts = result["internal_log"][-1].get("internal_thoughts", "")

    st.session_state.messages.append({
        "role": "assistant",
        "content": last_bot_msg,
        "thoughts": current_thoughts
    })

    # 4. Завершение интервью
    if result.get("finished", False):
        st.session_state.interview_active = False

        save_logs(result, filename=log_file, participant_name=name)

        st.session_state.messages.append({
            "role": "system",
            "content": "🏁 Интервью завершено. Спасибо за участие!"
        })

        st.toast("Интервью завершено", icon="🎉")
        st.balloons()

        with st.expander("📊 Итоговый фидбэк", expanded=True):
            st.markdown(result.get("final_feedback", "Фидбэк не найден"))

    st.rerun()
