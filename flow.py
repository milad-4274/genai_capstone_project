from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import os
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

SAVE_GRAPH_IMAGE = False

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# agent_instructions = ""
# for i, agent in enumerate([AccommodationSearchAgent,activity_search, destination_recommender,generate_itinerary, review_itinerary,silly_travel_stylist_structured,get_transportation,get_location_visa]):
#     print("Agent name: ", agent.__name__ , "Agent doc:",  agent.__doc__)
#     print("_"*10)


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
("system", """
🧠 SUPERVISOR AGENT PROMPT

ROLE:
You are the Supervisor Agent in a multi-agent travel planning system. You coordinate specialized agents, maintain an evolving shared context, and guide the user through a complete, intelligent trip-planning journey.

You do NOT generate travel content yourself — you only orchestrate agent calls and context updates.

🧰 ACCESS:
- `context`: the shared state (see variables below)
- all specialist agents (see formats below)
- `get_user_input`: to collect or confirm information, or deliver results

📦 CONTEXT VARIABLES:
transportation, chat_history, user_preferences, current_location, budget,
destination, visa_info, itinerary_draft, personalized_itinerary,
accommodation, duration, start_date, trip_tips

---

📦 AGENTS & INPUT FORMATS:

get_accommodation:
{{"destination": string, "user preference": string}}

activity_search:
{{"search_query": "a google search query that includes preferences of the user based on destination"}}

destination_recommender:
{{"travel_date": string, "duration": string, "budget": string, "accommodation_preferences": string}}

generate_itinerary:
{{
  "destination": string,
  "travel_date": string,
  "duration": string,
  "budget": string,
  "accommodation": string,
  "destination_info": string,
  "transportation": string,
  "trip_tips": string,
  "visa_info": string,
  "activity_preferences": "use both user preferences and destination_activities"
}}

silly_travel_stylist_structured:
{{"destination": string, "travel_date": string, "duration": string, "preferences": string, "budget": string}}

get_transportation:
{{"origin": string, "destination": string, "transportation_preferences": string, "start_date": string, "duration": string}}

get_location_visa:
{{"origin": string, "destination": string, "other": string}}

get_user_input:
Use this agent to gather missing info, confirm outputs, or deliver the final result.

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
     → Call `get_accommodation`.

5. 🚄 **Plan Transportation**
   - If `origin`, `destination`, `start_date`, and `duration` are present:
     → Call `get_transportation`.

6. 🧭 **Search for Activities**
   - If `duration` and `destination` are present:
     → Call `activity_search`.

7. 🎉 **Generate Fun Summary**
   - Only if `trip_tips` is NOT already in context
   - And if `duration`, `destination`, `travel_date`, `preferences`, and `budget` are present:
     → Call `silly_travel_stylist_structured`.

8. 🧳 **Generate Itinerary**
   - Only if `itinerary_draft` is NOT already in the current context
   - And if `destination`, `travel_date`, `duration`, `activity_preferences`,
     `budget`, `accommodation`, `destination_info`, `transportation`,
     `trip_tips`, `visa_info` are available:
     → Call `generate_itinerary`.

9. ✅ **Send Final Output to User**
   - If `itinerary_draft` is now available in the context instead of calling agent, do `action` to send the user the output
     → Return it using:
     ```json
     {{"action": "final_response", "response": the context itinerary_draft as json}}
     ```

---

⚠️ RULES:

- Only call an agent when all of its required inputs are available in `context`.
- If something is missing, use `get_user_input` to ask for it.
- Never generate travel content — delegate all logic to appropriate agents.
- Whenever you want to call an agent, return a JSON with key:
  ```json
  {{"why": "the reason you called this agent very briefly", "agent": "<name_of_the_agent>", "input": {{<agent_input>}} }}

"""),
("human", "Current context: {context}")
])
# 
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
        elif "action" in response:
            print("Final Result: \n", response["response"])
            return {"next_node": END}
            # return {"next_node": "final_response", "response": response["response"]}
        else:
            raise ValueError(f"Invalid supervisor response: {response}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON.  LLM Response: {result.content}")
        return {"next_node": "get_user_input", "agent_input": "I encountered an error. Please try again."}  # Or some error state
    except ValueError as e:
        print(f"ValueError: {e}")
        return {"next_node": "get_user_input", "agent_input": "I encountered an error processing the request."}


def get_user_input(state: TripState):
    print(" GET USER INPUT CALLED.")
    """Prompts the user for more information."""
    chat_history = state.get_chat_history()
    chat_history.append(state.agent_input)
    user_input = input("Model --- " + state.agent_input + "\nUser :\n")
    chat_history.append(user_input)
    new_context = context_summarizer(state, "\n".join(chat_history[-2:]))
    new_context.chat_history= chat_history
    new_context.response= user_input
    return new_context


def context_summarizer(state: TripState, text_input: str):
    # context summarizer
    """
    Updates the context (TripState) with new values extracted from text input using the LLM.
    Should be called after user input or agent output.
    """
    print("CONTEXT SUMMARIZER CALLED")
    context_str = str(state)
    context_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """
            You are a context summarizer. Your job is to update the travel planning context using newly received input.
            Look for these keys:
            user_preferences
            current_location
            budget
            destination
            visa_info
            personalized_itinerary
            duration
            start_date

            Return only those keys that are found or updated. Do NOT return unrelated or unchanged keys.

            Format:
            ```json
            {{"field1": "updated value", "field2": "another value"}}
            ```
        """),
        ("human", """
        Context so far:
        {context}

        New input:
        {text_input}
        """),
    ])
    
    chain = context_summary_prompt | llm_context
    result = chain.invoke({"context": context_str, "text_input": text_input})

    try:
        updates = extract_json_from_response(result.content)
        print("\nContext updates:", updates)
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)
        return state
    except Exception as e:
        print(f"❌ Failed to summarize context: {e}\nRaw output: {result.content}")
        return {"next_node": "get_user_input", "agent_input": "Something went wrong updating the context."}


# def final_response(state: TripState):
#     return {"response": f"Here's your personalized trip program for {state.destination}:\n\n{state.personalized_itinerary}\n\nVisa Information: {state.visa_eligibility}\nBudget Considerations: {state.budget}"}

# --- Router Nodes ---

def get_next_node(state: TripState):
    return state.next_node


def summarize_context_after_call(agent_func):
    def wrapper(state: TripState):
        print(f"⏳ Calling wrapped agent: {agent_func.__name__}")
        
        # Step 1: Call the original agent with state.agent_input
        output = agent_func(state.agent_input)

        # Step 2: Update the context based on agent output
        context_summarizer(state, json.dumps(output))  # Update in-place

        # Step 3: Optionally store response in state
        state.response = output

        return state
    return wrapper

@summarize_context_after_call
def agent_destination_wrapper(input_data):
    print("destination reccommender is called.")
    return destination_recommender(input_data)

@summarize_context_after_call
def get_location_visa_wrapper(input_data):
    return get_location_visa(input_data)

@summarize_context_after_call
def acc_agent_wrapper(input_data):
    print("Accomodation agent is called.")
    acc_agent = AccommodationSearchAgent()
    return acc_agent(input_data)

def silly_travel_stylist_structured_wrapper(state):
    print("I am called silly_travel_stylist_structured_wrapper")
    return {"trip_tips": silly_travel_stylist_structured(state.agent_input)}

def generate_itinerary_wrapper(state):
    print("generate itinerary is called.")
    result = generate_itinerary(state.agent_input)
    
    # store in state to signal the supervisor it's been done
    state.itinerary_draft = result
    return {"itinerary_draft": result}

@summarize_context_after_call
def get_transportation_wrapper(input_data):
    print("get transportation is called.")
    return get_transportation(input_data)

# @summarize_context_after_call
# def final_response_wrapper(input_data):
#     print("final response is called.")
#     return final_response(input_data)

def activity_search_wrapper(state:TripState):
    print("activity search is called.")
    return {"destination_activity": activity_search(state.agent_input)}

def start_node(state:TripState):
    state = context_summarizer(state, state.chat_history[0])
    print("state is ", state)
    return state


# Create the graph
graph = StateGraph(TripState)

graph.add_node("start", start_node)  # Start goes to supervisor
graph.add_node("supervisor", supervisor_node)  # Add the supervisor node
graph.add_node("get_user_input", get_user_input)
graph.add_node("activity_search", activity_search_wrapper)
graph.add_node("destination_recommender", agent_destination_wrapper)
graph.add_node("get_location_visa", get_location_visa_wrapper)
graph.add_node("get_accommodation", acc_agent_wrapper)
graph.add_node("silly_travel_stylist_structured", silly_travel_stylist_structured_wrapper)
graph.add_node("generate_itinerary", generate_itinerary_wrapper)
graph.add_node("get_transportation", get_transportation_wrapper)
# graph.add_node("final_response", final_response_wrapper)


# --- Edges ---
graph.add_edge("start", "supervisor")

graph.add_conditional_edges("supervisor", get_next_node)

graph.add_edge("get_user_input", "supervisor") # Go back to supervisor after user input
graph.add_edge("activity_search","supervisor")
graph.add_edge("destination_recommender","supervisor")
graph.add_edge("get_location_visa","supervisor")
graph.add_edge("get_transportation","supervisor")
graph.add_edge("get_accommodation","supervisor")
graph.add_edge("silly_travel_stylist_structured","supervisor")
graph.add_edge("generate_itinerary","supervisor")
# graph.add_edge("final_response","supervisor")
# Entrypoint of the graph
graph.set_entry_point("start")
# Compile the graph
chain = graph.compile()

if SAVE_GRAPH_IMAGE:
    from io import BytesIO
    from PIL import Image
    stream = BytesIO(chain.get_graph().draw_mermaid_png())
    image = Image.open(stream).convert("RGBA")
    image.save("flow_design.png")

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
    # inputs3 = {"chat_history": "Plan a trip to Vienna, Austria. I live in berlin and my budget is 2988"}
    # result3 = chain.invoke(inputs3)
    # print("\nResult 3:", result3) # Will likely ask for a budget

    # Example 4: Full information provided
    inputs4 = {"chat_history": "Plan a 7-day adventure trip to Bali for someone traveling from Seville with a budget of $2000. They enjoy hiking and beaches. The trip should start on April 19th 2025."}
    result4 = chain.invoke(inputs4)
    print("\nResult 4:", result4)
    
    
    
    pass