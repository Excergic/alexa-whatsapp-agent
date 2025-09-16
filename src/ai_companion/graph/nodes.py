from ai_companion import settings
from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_router_chain
from ai_companion.schedules.context_generation import ScheduleContextGenerator


def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context is not state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    return {"apply_activity": apply_activity, "current_activity": schedule_context}

async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke(
        {"messages" : state["messages"][-settings.ROUTER_MESSAGE_TO_ANALYZE :]}
    )
    return {"workflow": response.response_type}