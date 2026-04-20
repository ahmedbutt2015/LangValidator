from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    query: str
    conversation: List[dict]
    current_agent: str
    current_agent_output: Optional[str]
    validation_score: Optional[float]
    validation_feedback: Optional[str]
    retry_count: int
    max_retries: int
    halted: bool
    halt_reason: Optional[str]
    final_answer: Optional[str]
