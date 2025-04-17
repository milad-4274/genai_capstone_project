from typing import Dict, List, Tuple, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import os
# from agents import (AccommodationSearchAgent,
#                     activity_search,
#                     destination_recommender,
#                     generate_itinerary,
#                     review_itinerary,
#                     silly_travel_stylist_structured,
#                     get_transportation,
#                     get_location_visa,)

from agents.agent_accommodation import AccommodationSearchAgent
from agents.agent_activity import activity_search
from agents.agent_des_recom import destination_recommender
from agents.agent_itinerary_generator import generate_itinerary
from agents.agent_itinerary_reviewer import review_itinerary
from agents.agent_tip_gen import silly_travel_stylist_structured
from agents.agent_transportation import get_transportation
from agents.agent_visa import get_location_visa

from state import TripState
from dotenv.main import load_dotenv
import json
from utils import extract_json_from_response

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

agent_instructions = ""
for i, agent in enumerate([AccommodationSearchAgent,activity_search, destination_recommender,generate_itinerary, review_itinerary,silly_travel_stylist_structured,get_transportation,get_location_visa]):
    print("Agent name: ", agent.__name__ , "Agent doc:",  agent.__doc__)
    print("_"*10)


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
("system", """
🧠 SUPERVISOR AGENT PROMPT

ROLE:
You are the Supervisor Agent in a multi-agent travel planning system. You coordinate specialized agents, maintain an evolving shared context, and guide the user through a complete, intelligent trip-planning journey.

You do NOT generate travel content yourself — you only orchestrate agent calls and context updates.

🧰 ACCESS:
- `context`: the shared state (see variables below)
- all specialist agents (see formats below)
- `context_summarizer`: must be called after each user or agent response
- `get_user_input`: to collect or confirm information, or deliver results

📦 CONTEXT VARIABLES:
transportation, chat_history, user_preferences, current_location, budget,
destination, visa_info, itinerary_draft, personalized_itinerary,
accommodation, duration, start_date

---

📦 AGENTS & INPUT FORMATS:

AccommodationSearchAgent:
{
    "destination": string,
    "user preference": string
}

activity_search:
{
    "activity_preferences": string,
    "num_results": int,
    "how_many": int,
    "style": "friendly" | "formal" | "bullet"
}

destination_recommender:
{
    "travel_date": string,
    "duration": string,
    "budget": string,
    "accommodation_preferences": string (can include image URL)
}

generate_itinerary:
{
    "destination": string,
    "travel_date": string,
    "duration": string,
    "activity_preferences": string,
    "budget": string
}

review_itinerary:
{
    "destination": string,
    "travel_date": string,
    "duration": string,
    "activity_preferences": string,
    "budget": string,
    "itinerary": <itinerary_draft>
}

silly_travel_stylist_structured:
{
    "destination": string,
    "travel_date": string,
    "duration": string,
    "preferences": string,
    "budget": string
}

get_transportation:
{
    "origin": string,
    "destination": string,
    "transportation_preferences": string,
    "start_date": string,
    "duration": string
}

get_location_visa:
{
    "origin": string,
    "destination": string,
    "other": string (optional, e.g. nationality info)
}

get_user_input:
Use this agent to gather missing info, confirm outputs, or deliver the final result.

context_summarizer:
Use this to update the shared context after every interaction.

---

🎯 PLANNING FLOW:

1. 📝 **Collect Initial Preferences**
   - Ensure these exist in context: `user_preferences`, `current_location`, `budget`, `start_date`, `duration`
   - Otherwise, request them via `get_user_input`.

2. 📍 **Recommend Destination (Optional)**
   - If `destination` is missing or user is unsure:
     → Call `destination_recommender`.

3. 📜 **Check Visa Requirements**
   - If `origin` and `destination` are present and `visa_info` is missing:
     → Call `get_location_visa`.

4. 🛏️ **Search for Accommodation**
   - Only if `accommodation` is NOT already in context
   - And if `destination` and user accommodation preferences are available:
     → Call `AccommodationSearchAgent`.

5. 🚄 **Plan Transportation**
   - If `origin`, `destination`, `start_date`, and `duration` are present:
     → Call `get_transportation`.

6. 🧭 **Search for Activities**
   - When `activity_preferences` are known:
     → Call `activity_search`.

7. 🎉 **Generate Fun Summary (Optional)**
   - Optionally call `silly_travel_stylist_structured` to generate a personalized style summary.

8. 🧳 **Generate Itinerary**
   - Only when these are available:
     `destination`, `travel_date`, `duration`, `activity_preferences`, `budget`
   - → Call `generate_itinerary`.

9. 🔍 **Review Itinerary**
   - Once `itinerary_draft` is available:
     → Call `review_itinerary`.

10. ✅ **Send Final Output to User**
   - Deliver the reviewed itinerary using `get_user_input`
   - Optionally ask for feedback or final confirmation.

---

⚠️ RULES:

- Always call `context_summarizer` after each user input or agent result.
- Only call an agent when all of its required inputs are available in `context`.
- If something is missing, use `get_user_input` to ask for it.
- Never generate travel content — delegate all logic to appropriate agents.
"""),
("human", "Current context: {context}")
])

llm_supervisor = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
llm_context = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
# --- Worker Agent Definitions ---

# Supervisor Node
def supervisor_node(state: TripState):
    """
    This node acts as the supervisor, deciding which agent to call next.
    """
    context = str(state)
    chain = SUPERVISOR_PROMPT | llm_supervisor
    result = chain.invoke({"context": context})
    
    try:
        response = extract_json_from_response(result.content)
        print("response is ", response )
        if "agent" in response:
            agent_name = response["agent"]
            agent_input = response.get("input")  # Pass user input
            # if agent_name == "get_accommodation" and not state.accommodation:
            #     full_query = state.chat_history[-1]      # last user turn
            #     return {"next_node": agent_name, "agent_input": full_query}
            # elif agent_name != "get_accommodation":
            #     if isinstance(agent_input, str) and agent_input.strip().startswith("{"):
            #         try:
            #             agent_input = json.loads(agent_input.replace("'", '"'))  # handles single quotes
            #         except Exception as e:
            #             print("Agent input parsing failed:", e)
            return {"next_node": agent_name, "agent_input": agent_input}

            # else:
            #     # Skip accommodation call since it's already done
            #     return {"next_node": "supervisor"}  # Reinvoke supervisor or go to another fallback
        # elif "action" in response:
        #     print("Final Result: \n", response["response"])
        #     return {"next_node": END}
        #     # return {"next_node": "final_response", "response": response["response"]}
        else:
            raise ValueError(f"Invalid supervisor response: {response}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON.  LLM Response: {result.content}")
        return {"next_node": "get_user_input", "agent_input": "I encountered an error. Please try again."}  # Or some error state
    except ValueError as e:
        print(f"ValueError: {e}")
        return {"next_node": "get_user_input", "agent_input": "I encountered an error processing the request."}


def get_user_input(state: TripState):
    """Prompts the user for more information."""
    chat_history = state.get_chat_history()
    chat_history.append(state.agent_input)
    user_input = input("Model --- " + state.agent_input + "\nUser :\n")
    print("---")
    chat_history.append(user_input)
    # return {"response": user_input}
    return {"chat_history" : chat_history}


def context_summarizer(text_input):
    
    prompt = """
    based on the following text, update the provided context:
    input text:
    {text_input}
    
    old context:
    {context} 
    """
    # context summarizer
    print("CONTEXT SUMMARIZER CALLED")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are extraction system like regex for extracting information from texts. the Respond with a JSON object containing the following keys as information (if any): \n{{'destination : ... , budget: ..., current_location : ...'}}"),
        ("human", "Extract the information from the following user query if any: \nUser Query:\n {query}"),
    ])
    chain = prompt | llm_refinement
    result = chain.invoke({
        "query": state.chat_history[-2:],
    })
    
    try:
        response = extract_json_from_response(result.content)
        valid_response = {}
        for key, value in response.items():
            if hasattr(state, key):
                if bool(value):
                    valid_response[key] = value
                    print(f"state {key} updated")
            else:
                print(f"Warning: update information returned unknown key {key}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON.  LLM Response: {result.content}")
        return {"next_node": "final_response", "response": "I encountered an error. Please try again."}  # Or some error state
    except ValueError as e:
        print(f"ValueError: {e}")
        return {"next_node": "final_response", "response": "I encountered an error processing the request."}
    return valid_response


def final_response(state: TripState):
    return {"response": f"Here's your personalized trip program for {state.destination}:\n\n{state.personalized_itinerary}\n\nVisa Information: {state.visa_eligibility}\nBudget Considerations: {state.budget}"}

# --- Router Nodes ---

def get_next_node(state: TripState):
    return state.next_node

# Create the graph
graph = StateGraph(TripState)

graph.add_node("start", lambda state: {"next_node": "supervisor"})  # Start goes to supervisor
graph.add_node("supervisor", supervisor_node)  # Add the supervisor node
graph.add_node("user_input_step", get_user_input)
graph.add_node("update_information", update_information)
graph.add_node("destination_recommender", destination_recommender)
graph.add_node("get_location_visa", get_location_visa)
graph.add_node("review_itinerary", review_itinerary)
acc_agent = AccommodationSearchAgent()
graph.add_node("get_accommodation", acc_agent)
graph.add_node("silly_travel_stylist_structured", silly_travel_stylist_structured)
graph.add_node("generate_itinerary", generate_itinerary)
graph.add_node("get_transportation", get_transportation)
graph.add_node("final_response", final_response)


# --- Edges ---
graph.add_edge("start", "supervisor")

graph.add_conditional_edges("supervisor", get_next_node)

graph.add_edge("user_input_step", "supervisor") # Go back to supervisor after user input
graph.add_edge("update_information","supervisor")
graph.add_edge("destination_recommender","supervisor")
graph.add_edge("get_location_visa","supervisor")
graph.add_edge("get_transportation","supervisor")
graph.add_edge("get_accommodation","supervisor")
graph.add_edge("silly_travel_stylist_structured","supervisor")
graph.add_edge("generate_itinerary","supervisor")
graph.add_edge("review_itinerary","supervisor")
graph.add_edge("final_response","supervisor")
# Entrypoint of the graph
graph.set_entry_point("start")


# Compile the graph
chain = graph.compile()
# from io import BytesIO
# from PIL import Image
# stream = BytesIO(chain.get_graph().draw_mermaid_png())
# image = Image.open(stream).convert("RGBA")
# image.save("flow_design.png")

# --- Example Usage ---
if __name__ == "__main__":
    # # Example 1: Initial input provides all information
    # inputs1 = {"chat_history": "I want to plan a 5-day trip to Paris. I'm interested in museums and good food, and my budget is around $1500."}
    # result1 = chain.invoke(inputs1)
    # print("\nResult 1:", result1)

    # # # Example 2: Initial input missing destination
    # inputs2 = {"chat_history": "I want to plan a trip. What are some popular destinations?"}
    # result2 = chain.invoke(inputs2)
    # print("\nResult 2:", result2) # Will likely ask for a destination

    # Example 3: Initial input with destination but missing budget
    inputs3 = {"chat_history": "Plan a trip to Vienna, Austria. I live in berlin and my budget is 2988"}
    result3 = chain.invoke(inputs3)
    print("\nResult 3:", result3) # Will likely ask for a budget

    # Example 4: Full information provided
    # inputs4 = {"chat_history": "Plan a 7-day adventure trip to Bali for someone traveling from Seville with a budget of $2000. They enjoy hiking and beaches."}
    # result4 = chain.invoke(inputs4)
    # print("\nResult 4:", result4)
    
    
    
    pass