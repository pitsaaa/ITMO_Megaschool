import operator
from typing import Annotated, List, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class InterviewState(TypedDict):
    # Сообщения накапливаются (это правильно)
    messages: Annotated[List[BaseMessage], operator.add]
    
    candidate_info: Dict[str, str]
    
    # Темы накапливаются (чтобы не повторяться)
    topics_covered: Annotated[List[str], operator.add]
    
    # Данные внутри хода (перезаписываются каждый раз)
    observer_analysis: Optional[Dict[str, Any]]
    expert_plan: Optional[Dict[str, Any]]
    
    # ВАЖНО: Убрали operator.add! 
    # Теперь Observer будет создавать чистый лист, а Expert — дополнять его вручную.
    current_turn_thoughts: List[str]
    
    # Итоговый лог накапливается
    internal_log: Annotated[List[Dict[str, Any]], operator.add]
    last_bot_msg: str
    
    final_feedback: Optional[Dict[str, Any]]
    finished: bool
    