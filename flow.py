from typing import Dict, List, Tuple, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import os
from agents import (AccommodationSearchAgent,
                    agent_activity)
from state import TripState
from dotenv.main import load_dotenv
import json
from utils import extract_json_from_response

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# prompt_v1 = [
#     ("system", """You are a trip planning supervisor agent. Your role is to manage the overall trip planning process. Give the value named `Current context` follow the instructions to generate personalized itinerary. The overall process include extracting user preferences and information like user current location, destination and desired activities. the user current location is required and if not provided, ask user to complete using `user_input_step` agent, destination and activities and budget are required too unless user wants a totally blind trip. Convert user output to structured context for updating current location destination and budget using `update_information` agent; Call it when some information exist in chat history but not in the context keys. if user input is not okay, ask them. If you have question from user call the `user_input_step` agent and ask follow-up question from user. If user is unsure about destination provide suggestions based on the current location and desired activities.
#     when the destination is set, if it is abroad, find about visa eligibility using `get_location_visa` agent. then research more about destination and try to generate itinerary and refine it based on user preferences in the context then respond using `action : final_response` instead of calling another agent to finish the process. in follow up questions try to suggest some options based on the context.
     
     
#     by deciding which agent to call next, or when to respond to the user.
    
#     Here are the available agents:
#     - extract_preferences: Extracts the user's travel preferences, interests, and style.
#     - get_location_visa:  Determines visa requirements.
#     - research_destination:  Researches a destination and provides information.
#     - plan_budget:  Provides a budget breakdown for a trip.
#     - generate_itinerary:  Generates a draft itinerary.
#     - refine_itinerary: Refines the itinerary based on user feedback.
#     - final_response:  Provides the final trip plan to the user.
#     - user_input_step: Asks the user for more input
#     - update_information: update budget, current location and destination. should be called after getting these information from user to register them.

#     Respond with a JSON object in the following format to call an agent:
#     - {{ "agent": "agent_name", "input": agent_input }}
    
#     If you're satisfied with the refined itinerary output respond in the following format to the user. 
#     - {{ "action": "final_response", "response": "final version of personalized itinerary" }}
#     """),
#     ("human", "Current context: {context}"),]
# --- Prompts ---
SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
("system", """
You are a *trip planning supervisor agent*. Your role is to orchestrate the end-to-end trip planning process by intelligently coordinating a set of specialized agents.

Your main goal is to deliver a **final personalized itinerary** to the user based on their inputs and preferences.

---

## 🎯 PROCESS OVERVIEW

The planning flow typically includes:
1. Extracting the user's travel preferences.
2. Collecting key trip information: `current_location`, `destination`, and `budget`.
3. Assisting the user if they're unsure (e.g., suggesting destinations).
4. Updating and validating the context.
5. Researching accommodations for the given destination or query.
6. Researching visa eligibility and destination details.
7. Generating and refining an itinerary.
8. Providing a final personalized response.

**IMPORTANT**: 
- Do **not** skip `get_accommodation` if `destination` and `budget` are available. 
- This is mandatory before generating the final itinerary.
---

## 🧠 CONTEXT USAGE

The system maintains a `Current context`, which includes:
- `chat_history`: chat history
- `user_preferences`: user_preferences
- `current_location`: current_location
- `destination`: destination
- `visa_eligibility`: visa_eligibility
- `budget`: budget
- `destination_info`: destination_info
- `itinerary_draft`: itinerary_draft
- `personalized_itinerary`: personalized_itinerary
- `next_node`: next_node
- `agent_input`: agent_input
- `response`: response
- `accommodation`: accommodation

---

## 🤖 AGENT INSTRUCTIONS

You have access to the following agents:

- **extract_preferences**: Extracts user's travel style and interests.
- **get_location_visa**: Checks visa requirements based on nationality and destination.
- **research_destination**: Gathers info about the destination.
- **get_accommodation**: will recommend hotels in a city.
- **plan_budget**: Provides a suggested trip budget breakdown.
- **generate_itinerary**: Creates a draft itinerary.
- **refine_itinerary**: Improves the itinerary based on feedback.
- **final_response**: Sends the final itinerary to the user.
- **user_input_step**: Asks the user for any needed input.
- **update_information**: Updates the context with `current_location`, `destination`, and/or `budget`.

---


## 🙋 USER INTERACTIONS

- If a required field is missing (e.g. `current_location`), ask the user using `user_input_step`.
- If the user is unsure about their destination, provide suggestions based on their `current_location` and `user_preferences`.
- Always confirm unclear or ambiguous user responses before updating the context.

---
## 🔁 WHEN TO USE `update_information`

Call `update_information` **only when**:
❗ Call update_information only if one or more of these fields are missing in context AND clearly available in chat history.
✅ If a required field is missing in context and not extractable from chat history, use user_input_step to ask the user directly.
🚫 Never call update_information with values that already exist in context.

---

## ✨ Refining the Itinerary

If an itinerary is generated and you have:
- Specific user feedback from `chat_history`, or
- Clear preference updates in `user_preferences`,

...then you may call the `refine_itinerary` agent ONCE to improve the itinerary.

✅ Only call `refine_itinerary` one time unless new feedback is added.
🚫 Do NOT call it repeatedly without user input or significant changes.
🤝 If the initial itinerary already matches user preferences well enough, proceed to `final_response`.

---

### 🚀 When to Respond with Final Itinerary

Once the following are present:
- A reasonable itinerary is generated (`itinerary_draft` or `personalized_itinerary`),
- Required trip information is filled (`destination`, `current_location`, `budget`, `user_preferences`),

...and there's no major user objection:

✅ Respond directly using:
```json
{{ "action": "final_response", "response": "final personalized itinerary" }}
```


## ✅ FINAL OBJECTIVE

Once the context has:
- `current_location`
- `destination`
- `budget`
- `user_preferences`

...you should:
1. Check visa eligibility if traveling abroad (`get_location_visa`).
2. Research the destination (`research_destination`).
3. Generate and refine the itinerary (`generate_itinerary`, `refine_itinerary`).
4. Respond to the user using:
```json
{{ "action": "final_response", "response": "final personalized itinerary here" }}
```
## 🧾 RESPONSE FORMAT
To call an agent:

```json
{{ "agent": "agent_name", "input": agent_input }}
```

To respond with the final plan:
```json
{{ "action": "final_response", "response": "your final trip plan" }}

Before the json, tell the reason of your selection in one line.
```
"""),
("human", "Current context: {context}")
])




llm_supervisor = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
llm_preference = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_location_visa = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
llm_destination = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_budget = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)
llm_itinerary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_refinement = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
llm_user_input = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)

# --- Worker Agent Definitions ---

# Supervisor Node
def supervisor_node(state: TripState):
    """
    This node acts as the supervisor, deciding which agent to call next.
    """

    context = str(state)
    
    # print("context\n", context)
    
    
    chain = SUPERVISOR_PROMPT | llm_supervisor
    result = chain.invoke({"context": context})
    
    try:
        response = extract_json_from_response(result.content)
        print("response is ", response )
        if "agent" in response:
            agent_name = response["agent"]
            agent_input = response.get("input")  # Pass user input
            if agent_name == "get_accommodation" and not state.accommodation:
                full_query = state.chat_history[-1]      # last user turn
                return {"next_node": agent_name, "agent_input": full_query}
            elif agent_name != "get_accommodation":
                return {"next_node": agent_name, "agent_input": agent_input}
            else:
                # Skip accommodation call since it's already done
                return {"next_node": "supervisor"}  # Reinvoke supervisor or go to another fallback
        elif "action" in response:
            print("Final Result: \n", response["response"])
            return {"next_node": END}
            # return {"next_node": "final_response", "response": response["response"]}
        else:
            raise ValueError(f"Invalid supervisor response: {response}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON.  LLM Response: {result.content}")
        return {"next_node": "final_response", "response": "I encountered an error. Please try again."}  # Or some error state
    except ValueError as e:
        print(f"ValueError: {e}")
        return {"next_node": "final_response", "response": "I encountered an error processing the request."}


def get_user_input(state: TripState):
    """Prompts the user for more information."""
    chat_history = state.get_chat_history()
    chat_history.append(state.agent_input)
    user_input = input("Model --- " + state.agent_input + "\nUser :\n")
    print("---")
    chat_history.append(user_input)
    # return {"response": user_input}
    return {"chat_history" : chat_history}

def extract_preferences(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the user's travel preferences, interests, and style from their input."),
        ("human", "{user_input}"),
    ])
    chain = prompt | llm_preference
    result = chain.invoke({"user_input": state.chat_history})
    return {"user_preferences": result.content}

def get_location_visa(state: TripState):
    duration = state.duration if hasattr(state, "duration") else "One week"
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with knowledge of Visa requirements by nationality for tourism purposes."),
        ("human", "For a {duration} trip from {current_location} to {destination} what is visa requirement. give your estimation of time for processing the probable application, the recommeded visa type and the cost of recommended visa "),
        # ("human", "{user_input}"),
    ])
    chain = prompt | llm_location_visa
    result = chain.invoke({"current_location": state.current_location, "destination": state.destination, "duration" : duration})
    return {"visa_eligibility": result.content, "current_location": state.current_location}

def research_destination(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Research '{destination}' and provide key information about attractions, local culture, and things to do."),
        ("human", "Tell me about {destination}."),
    ])
    chain = prompt | llm_destination
    result = chain.invoke({"destination": state.destination})
    return {"destination_info": result.content}

def plan_budget(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Consider the user's budget '{budget}' and the destination. Provide a rough breakdown of potential costs for a trip to '{destination}'."),
        ("human", "How can I plan a trip to {destination} with a budget of {budget}?"),
    ])
    chain = prompt | llm_budget
    result = chain.invoke({"destination": state.destination, "budget": state.budget})
    return {"budget": state.budget}

def generate_itinerary(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the user's preferences '{user_preferences}', information about '{destination_info}', and budget considerations, generate a draft itinerary."),
        ("human", "Create a potential itinerary."),
    ])
    chain = prompt | llm_itinerary
    result = chain.invoke({
        "user_preferences": state.user_preferences,
        "destination_info": state.destination_info,
        "budget": state.budget
    })
    return {"itinerary_draft": result.content}



def refine_itinerary(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Refine the itinerary based on the user's initial input and preferences to make it more personalized. Only return the Markdown result of the provided itinerary without any further explantion and extra texts."),
        ("human", "**User Preferences: ** \n{user_input}\n \n **Draft itinerary:** {itinerary_draft}"),
    ])
    chain = prompt | llm_refinement
    itinerary = state.personalized_itinerary if bool(state.personalized_itinerary) else state.itinerary_draft
    result = chain.invoke({
        "user_input": state.user_preferences,
        "itinerary_draft": itinerary
    })
    return {"personalized_itinerary": result.content}


def update_information(state: TripState):
    print("UPDATE INFORMATION CALLED")
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
graph.add_node("extract_preferences", extract_preferences)
graph.add_node("get_location_visa", get_location_visa)
graph.add_node("research_destination", research_destination)
acc_agent = AccommodationSearchAgent()
graph.add_node("get_accommodation", acc_agent)
graph.add_node("plan_budget", plan_budget)
graph.add_node("generate_itinerary", generate_itinerary)
graph.add_node("refine_itinerary", refine_itinerary)
graph.add_node("final_response", final_response)


# --- Edges ---
graph.add_edge("start", "supervisor")

graph.add_conditional_edges("supervisor", get_next_node)

graph.add_edge("user_input_step", "supervisor") # Go back to supervisor after user input
graph.add_edge("update_information","supervisor")
graph.add_edge("extract_preferences","supervisor")
graph.add_edge("get_location_visa","supervisor")
graph.add_edge("research_destination","supervisor")
graph.add_edge("get_accommodation","supervisor")
graph.add_edge("plan_budget","supervisor")
graph.add_edge("generate_itinerary","supervisor")
graph.add_edge("refine_itinerary","supervisor")
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