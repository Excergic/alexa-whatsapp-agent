from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion import settings

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import get_character_response_chain, get_router_chain
from ai_companion.graph.utils.helpers import get_chat_model

from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager

# why context_injection_node?
# because we need to inject the current activity into the character card to make alexa a real person
def context_injection_node(state: AICompanionState):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""): # != and is not are not the same
        apply_activity = True
    else:
        apply_activity = False

    return {"apply_activity": apply_activity, "current_activity": schedule_context}

# node will decide which workflow to use based on the current activity, text, image or audio
async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke(
        {"messages" : state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE :]}
    )
    return {"workflow": response.response_type}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke({
        "messages" : state["messages"], # for conversation history if Missing: Alexa will not have not past memory of chat
        "current_activity" : current_activity,
        "memory_context" : memory_context
    }, config)

    return {"messages" : AIMessage(content=response)}

async def summarize_conversation_node(state: AICompanionState):
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary: 
        summary_message = (
            f"This is summary of the conversation to date between Alexa and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}

async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""

    if not state["messages"]:
        return {}
    
    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}

async def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}

