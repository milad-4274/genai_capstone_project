from langchain_core.messages import BaseMessage
from typing import List


# --- State Definition ---
class TripState:
    chat_history: List[str] = None
    user_preferences: str = None
    current_location: str = None
    destination: str = None
    accommodation: str = None
    visa_eligibility: str = None
    budget: str = None
    destination_info: str = None
    itinerary_draft: str = None
    personalized_itinerary: str = None
    messages: List[BaseMessage]
    next_node: str = None # Added for supervisor
    agent_input: str = None # Added for supervisor
    response: str = None # Added for final response

    def __init__(self, chat_history: str, **kwargs):
        self.chat_history = [chat_history]
        self.user_preferences = kwargs.get("user_preferences", chat_history)
        self.current_location = kwargs.get("current_location")
        self.destination = kwargs.get("destination")
        self.accommodation = kwargs.get('accommodation')
        self.visa_eligibility = kwargs.get("visa_eligibility")
        self.budget = kwargs.get("budget")
        self.destination_info = kwargs.get("destination_info")
        self.itinerary_draft = kwargs.get("itinerary_draft")
        self.personalized_itinerary = kwargs.get("personalized_itinerary")
        self.next_node = kwargs.get("next_node") # Added
        self.agent_input = kwargs.get("agent_input") # Added
        self.response = kwargs.get("response")
    
    def get_chat_history(self):
        # print("Appending to Chat history", response)
        # self.chat_history.append(response)
        return self.chat_history
    
    def __repr__(self):
        return f"""
    chat_history : {self.chat_history}
    user_preferences : {self.user_preferences}
    current_location : {self.current_location}
    destination : {self.destination}
    accommodation: {self.accommodation}
    visa_eligibility : {self.visa_eligibility}
    budget : {self.budget}
    destination_info : {self.destination_info}
    itinerary_draft : {self.itinerary_draft}
    personalized_itinerary : {self.personalized_itinerary}
    next_node : {self.next_node}
    agent_input : {self.agent_input}
    response : {self.response}
    """
    
    def __str__(self):
        return f"""
    chat_history : {self.chat_history}
    user_preferences : {self.user_preferences}
    current_location : {self.current_location}
    destination : {self.destination}
    accommodation: {self.accommodation}
    visa_eligibility : {self.visa_eligibility}
    budget : {self.budget}
    destination_info : {self.destination_info}
    itinerary_draft : {self.itinerary_draft}
    personalized_itinerary : {self.personalized_itinerary}
    next_node : {self.next_node}
    agent_input : {self.agent_input}
    response : {self.response}
    """