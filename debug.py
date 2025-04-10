import os
from dotenv.main import load_dotenv
from IPython.display import Image, display
from pprint import pprint


from typing import Annotated
from typing_extensions import TypedDict
from typing import Literal

from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import ToolNode



from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from collections.abc import Iterable
from random import randint

from langchain_core.messages.tool import ToolMessage


from langchain_core.tools import tool
from typing import List, Dict, Optional
import requests
import json
import traceback

from io import BytesIO
from PIL import Image


from prompts import TRIPADVISOR_SYSINT, WELCOME_MSG

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID= os.getenv("GOOGLE_CSE_ID")
GOOGLE_SEARCH_KEY= os.getenv("GOOGLE_SEARCH_KEY")

for env_var in ["GOOGLE_API_KEY", "GOOGLE_CSE_ID", "GOOGLE_SEARCH_KEY"]:
    if env_var is None:
        raise ValueError(f"{env_var} must be set")

class SearchState(TypedDict):
    """State representing the customer's search conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress search.
    search: list[str]

    # Flag indicating that the search is placed and completed.
    finished: bool


google_search_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
supervisor_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# The default recursion limit for traversing nodes is 25 - setting it higher means
# you can try a more complex search with multiple steps and round-trips (and you
# can chat for longer!)
config = {"recursion_limit": 25}


def chatbot(state: SearchState) -> SearchState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    message_history = [TRIPADVISOR_SYSINT] + state["messages"]
    return {"messages": [supervisor_llm.invoke(message_history)]}


def human_node(state: SearchState) -> SearchState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}

def maybe_exit_human_node(state: SearchState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"



@tool("google_web_search")
def google_web_search(
    query: str,
    num_results: int = 5,      
    how_many: int = 5,            
    style: str = "friendly",    
    **kwargs,
) -> str:
    """
    Search Google (Programmable Search Engine) and return a **summarised list**
    of the best places/attractions for the user.

    Args:
        query: Natural‑language search, e.g. "best museums in Paris".
        num_results: Raw results fetched from Google (max 10).
        how_many: Items to keep after LLM re‑ranking.
        style: Output tone ("friendly", "formal", "bullet").
    """
    if not (GOOGLE_SEARCH_KEY and GOOGLE_CSE_ID):
        raise RuntimeError(
            "Set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID env vars first."
        )

    # Google Search API ───────────────────────────
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": min(num_results, 5),
    }

    try:
        raw = requests.get(url, params=params, timeout=10).json()
        items: List[Dict] = raw.get("items", [])
        # titles = [item.get("title") for item in items]  
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Google Search error: {e}")

    if not items:
        return "I couldn't find any results for that search."

    # preparing prompt for LLM ────────────────────────────
    compact = [{"title": it["title"], "link": it["link"], "snippet": it["snippet"]} for it in items]

    prompt = (
        f"You are a travel expert.\n"
        f"Pick the top {how_many} places someone should visit based on the "
        f"Google search results below and present them in a {style} style. "
        "Give each place a one‑sentence description.\n\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )
    
    # print("search summary", compact)
    # print("items", items[0])

    # summerization with LLM ───────────────────────────────────
    summary = google_search_llm.invoke(prompt).content
    # Return as a dictionary with a tag (or using your message format) so that the router knows it is from tool.
    return {"content": summary, "from": "tool"}




# Define the tools and create a "tools" node.
tools = [google_web_search]
tool_node = ToolNode(tools)

# Attach the tools to the model so that it knows what it can call.
google_search_llm = google_search_llm.bind_tools(tools)


def maybe_route_to_tools(state: SearchState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    # Force human route if only the welcome message exists
    if len(msgs) == 1:
        return "human"
    
    msg = msgs[-1]

    if state.get("finished", False):
        # When an search is placed, exit the app. The system instruction indicates
        # that the chatbot should say thanks and goodbye at this point, so we can exit
        # cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "searching"

    else:
        return "human"
    
def chatbot_with_tools(state: SearchState) -> SearchState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"search": [], "finished": False}

    if state["messages"]:
        new_output = google_search_llm.invoke([TRIPADVISOR_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}



def search_node(state: SearchState) -> SearchState:
    """The searching node. This is where the search state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    search = state.get("search", [])
    outbound_msgs = []
    search_placed = False

    for tool_call in tool_msg.tool_calls:

        if tool_call["name"] == "google_web_search":

            # Each search item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            search.append(f'{tool_call["args"]["drink"]} ({modifier_str})')
            response = "\n".join(search)

        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "search": search, "finished": search_placed}

graph_builder = StateGraph(SearchState)

# Add the nodes, including the new tool_node.
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("searching", search_node)

# Chatbot -> {searching, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("searching", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_menu = graph_builder.compile()



stream = BytesIO(graph_with_menu.get_graph().draw_mermaid_png())
image = Image.open(stream).convert("RGBA")
image.save("final_graph.png")

# image = Image(graph_with_menu.get_graph().draw_mermaid_png())

# state = graph_with_menu.invoke({"messages": []}, config)

