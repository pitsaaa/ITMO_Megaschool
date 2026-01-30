import json
import sys
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # Важно для логов

# Твои импорты
from utils.state import InterviewState
from agents.observer import observer_node
from agents.expert import expert_node
from agents.interviewer import interviewer_node
from agents.feedback import feedback_node

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def route_signal(state: InterviewState):
    if state.get("finished", False):
        return "feedback"
    return END



def build_graph():
    workflow = StateGraph(InterviewState)
    workflow.add_node("observer", observer_node)
    workflow.add_node("expert", expert_node)
    workflow.add_node("interviewer", interviewer_node)
    workflow.add_node("feedback", feedback_node)

    workflow.set_entry_point("observer")
    workflow.add_edge("observer", "expert")
    workflow.add_edge("expert", "interviewer")
    
    workflow.add_conditional_edges(
        "interviewer",
        route_signal,
        {"feedback": "feedback", END: END}
    )
    workflow.add_edge("feedback", END)

    shared_memory = MemorySaver()
    return workflow.compile(checkpointer=shared_memory)

def save_logs(state: InterviewState, filename="interview_log.json", participant_name="Candidate"):
    feedback_text = state.get("final_feedback", "Feedback not generated")
    final_data = {
        "participant_name": participant_name,
        "turns": state.get("internal_log", []),
        "final_feedback": feedback_text 
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print(f"\n{Colors.WARNING}📁 Лог сохранен в {filename}{Colors.ENDC}")

def main():
    app = build_graph()
    # Нужен для работы checkpointer
    config = {"configurable": {"thread_id": "interview_1"}}
    
    print(f"{Colors.HEADER}=== AI INTERVIEW SYSTEM V2 ==={Colors.ENDC}")
    
    name = input("ФИО: ") or "Ivanov Ivan"
    role = input("Позиция: ") or "C++ Developer"
    level = input("Грейд: ") or "Middle"
    stack = input("Стек: ") or "C++, Postgres"
    log_filename = input("Имя файла лога: ") or "interview_log.json"

    print(f"\n{Colors.WARNING}🚀 Начало...{Colors.ENDC}\n")
    
    # ПЕРВЫЙ ШАГ: Бот здоровается и задает вопрос
    # Мы не чистим лог после этого!
    result = app.invoke({
        "messages": [HumanMessage(content="Начни интервью.")],
        "candidate_info": {"name": name, "role": role, "level": level, "stack": stack},
        "topics_covered": [],
        "internal_log": [],
        "finished": False
    }, config=config)

    print(f"{Colors.GREEN}Interviewer:{Colors.ENDC} {result['messages'][-1].content}")

    while True:
        user_text = input(f"\n{Colors.BOLD}You:{Colors.ENDC} ")
        
        # Запускаем граф. Благодаря thread_id он сам достанет старый state
        result = app.invoke({
            "messages": [HumanMessage(content=user_text)]
        }, config=config)
        
        # Вывод мыслей
        if result.get("current_turn_thoughts"):
            print(f"\n{Colors.CYAN}" + "\n".join(result["current_turn_thoughts"]) + f"{Colors.ENDC}")

        # Вывод ответа
        print(f"\n{Colors.GREEN}Interviewer:{Colors.ENDC} {result['messages'][-1].content}")
        
        if result.get("finished", False):
            save_logs(result, filename=log_filename, participant_name=name)
            break

if __name__ == "__main__":
    main()