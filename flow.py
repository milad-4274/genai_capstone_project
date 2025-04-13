from typing import Dict, List, Tuple, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
import os
from dotenv.main import load_dotenv
import json
from utils import extract_json_from_response

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")



# --- Prompts ---
SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a trip planning supervisor agent. Your role is to manage the overall trip planning process. The overall process include extracting user preferences and information like user current location, destination and desired activities. register them using update_information agent to help you to see in the context. If you have question from user call the `user_input_step` agent and ask follow-up question from user. If user is unsure about destination provide suggestions based on the current location and desired activities. when the destination is set, if it is abroad, find about visa eligibility using `get_location_visa` agent. then research more about destination and try to generate itinerary and refine it based on user preferences in the context then respond using `action : final_response` instead of calling another agent to finish the process.
     
     
    by deciding which agent to call next, or when to respond to the user.
    
    Here are the available agents:
    - extract_preferences: Extracts the user's travel preferences, interests, and style.
    - get_location_visa:  Determines visa requirements.
    - research_destination:  Researches a destination and provides information.
    - plan_budget:  Provides a budget breakdown for a trip.
    - generate_itinerary:  Generates a draft itinerary.
    - refine_itinerary: Refines the itinerary based on user feedback.
    - final_response:  Provides the final trip plan to the user.
    - user_input_step: Asks the user for more input
    - update_information: update budget, current location and destination. should be called after getting these information from user to register them.

    Respond with a JSON object in the following format to call an agent:
    - {{ "agent": "agent_name", "input": agent_input }}
    
    If you're satisfied with the refined itinerary output respond in the following format to the user. 
    - {{ "action": "final_response", "response": "final version of personalized itinerary" }}
    """),
    ("human", "{user_input}  Current context: {context}"),
])




llm_supervisor = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
llm_preference = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_location_visa = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
llm_destination = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_budget = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)
llm_itinerary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)
llm_refinement = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
llm_user_input = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.5)

# --- State Definition ---
class TripState:
    user_input: str
    user_preferences: str = None
    current_location: str = None
    destination: str = None
    visa_eligibility: str = None
    budget: str = None
    destination_info: str = None
    itinerary_draft: str = None
    personalized_itinerary: str = None
    messages: List[BaseMessage]
    next_node: str = None # Added for supervisor
    agent_input: str = None # Added for supervisor
    response: str = None # Added for final response

    def __init__(self, user_input: str, **kwargs):
        self.user_input = [user_input]
        self.user_preferences = kwargs.get("user_preferences", user_input)
        self.current_location = kwargs.get("current_location")
        self.destination = kwargs.get("destination")
        self.visa_eligibility = kwargs.get("visa_eligibility")
        self.budget = kwargs.get("budget")
        self.destination_info = kwargs.get("destination_info")
        self.itinerary_draft = kwargs.get("itinerary_draft")
        self.personalized_itinerary = kwargs.get("personalized_itinerary")
        self.messages = kwargs.get("messages", [])
        self.next_node = kwargs.get("next_node") # Added
        self.agent_input = kwargs.get("agent_input") # Added
        self.response = kwargs.get("response")
    
    def __repr__(self):
        return f"""
    user_input : {self.user_input}
    user_preferences : {self.user_preferences}
    current_location : {self.current_location}
    destination : {self.destination}
    visa_eligibility : {self.visa_eligibility}
    budget : {self.budget}
    destination_info : {self.destination_info}
    itinerary_draft : {self.itinerary_draft}
    personalized_itinerary : {self.personalized_itinerary}
    messages : {self.messages}
    next_node : {self.next_node}
    agent_input : {self.agent_input}
    response : {self.response}
    """
# --- Worker Agent Definitions ---

# Supervisor Node
def supervisor_node(state: TripState):
    """
    This node acts as the supervisor, deciding which agent to call next.
    """
    context = {
        "user_preferences": state.user_preferences,
        "destination": state.destination,
        "visa_eligibility": state.visa_eligibility,
        "budget": state.budget,
        "destination_info": state.destination_info,
        "itinerary_draft": state.itinerary_draft,
        "current_location": state.current_location
    }
    
    # print("Calling Supervisor node, state is \n", state)
    
    chain = SUPERVISOR_PROMPT | llm_supervisor
    result = chain.invoke({"user_input": state.user_input[-1], "context": context})
    
    try:
        response = extract_json_from_response(result.content)
        print("response is ", response )
        if "agent" in response:
            agent_name = response["agent"]
            agent_input = response.get("input")  # Pass user input
            return {"next_node": agent_name, "agent_input": agent_input}
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
    prompt = ChatPromptTemplate.from_messages([
        ("system", "The user's request is incomplete. Ask a clarifying question."),
        ("human", "{user_input}"),
        ("system", "Respond with a single question to get the missing information."),
    ])
    chain = prompt | llm_user_input
    user_input = input("Model --- " + state.agent_input + "User :\n")
    print("---")
    result = chain.invoke({"user_input": user_input})
    return {"response": result.content}

def extract_preferences(state: TripState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the user's travel preferences, interests, and style from their input."),
        ("human", "{user_input}"),
    ])
    chain = prompt | llm_preference
    result = chain.invoke({"user_input": state.user_input})
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
    result = chain.invoke({
        "user_input": state.user_input,
        "itinerary_draft": state.itinerary_draft
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
        "query": state.user_input[-1],
    })
    
    try:
        response = extract_json_from_response(result.content)
        valid_response = {}
        for key, value in response.items():
            if hasattr(state, key):
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
graph.add_edge("plan_budget","supervisor")
graph.add_edge("generate_itinerary","supervisor")
graph.add_edge("refine_itinerary","supervisor")
graph.add_edge("final_response","supervisor")
# Entrypoint of the graph
graph.set_entry_point("start")

# Conditional edges for initial setup
# graph.add_conditional_edges(
#     "start",  # Start from the 'start' node
#     should_get_preferences,
#     {True: "extract_preferences", False: "check_destination"}
# )


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
    # inputs1 = {"user_input": "I want to plan a 5-day trip to Paris. I'm interested in museums and good food, and my budget is around $1500."}
    # result1 = chain.invoke(inputs1)
    # print("\nResult 1:", result1)

    # # Example 2: Initial input missing destination
    # inputs2 = {"user_input": "I want to plan a trip. What are some popular destinations?"}
    # result2 = chain.invoke(inputs2)
    # print("\nResult 2:", result2) # Will likely ask for a destination

    # # Example 3: Initial input with destination but missing budget
    # inputs3 = {"user_input": "Plan a trip to Rome."}
    # result3 = chain.invoke(inputs3)
    # print("\nResult 3:", result3) # Will likely ask for a budget

    # Example 4: Full information provided
    inputs4 = {"user_input": "Plan a 7-day adventure trip to Bali for someone traveling from Seville with a budget of $2000. They enjoy hiking and beaches."}
    result4 = chain.invoke(inputs4)
    print("\nResult 4:", result4)
    
    
    
    
    pass