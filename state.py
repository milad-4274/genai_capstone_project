from langchain_core.messages import BaseMessage
from typing import List


# --- State Definition ---
class TripState:
    chat_history: List[str] = None
    user_preferences: str = None
    current_location: str = None
    budget: str = None
    destination: str = None
    visa_info: str = None
    itinerary_draft: str = None
    personalized_itinerary: str = None
    accommodation: str = None
    duration : str = None
    start_date : str = None
    trip_tips : str = None
    destination_activity: str = None
    transportation : str = None
    next_node: str = None # Added for supervisor
    agent_input: str = None # Added for supervisor
    
    def __init__(self, chat_history: str, **kwargs):
        self.chat_history = [chat_history]
        self.user_preferences = kwargs.get("user_preferences", chat_history)
        self.current_location = kwargs.get("current_location")
        self.budget = kwargs.get("budget")
        self.destination = kwargs.get("destination")
        self.visa_info = kwargs.get("visa_info")
        self.itinerary_draft = kwargs.get("itinerary_draft")
        self.personalized_itinerary = kwargs.get("personalized_itinerary")
        self.accommodation = kwargs.get("accommodation")
        self.duration = kwargs.get("duration")
        self.start_date = kwargs.get("start_date")
        self.trip_tips = kwargs.get("trip_tips")
        self.destination_activity = kwargs.get("destination_activity")
        self.transportation = kwargs.get("transportation")
        self.next_node = kwargs.get("next_node")
        self.agent_input = kwargs.get("agent_input")
        # self.response = kwargs.get("response").
        
    def get_chat_history(self):
        # print("Appending to Chat history", response)
        # self.chat_history.append(response)
        return self.chat_history
    
    def __repr__(self):
        return f"""
    chat_history : {self.chat_history}
    user_preferences : {self.user_preferences}
    current_location : {self.current_location}
    budget : {self.budget}
    destination : {self.destination}
    visa_info : {self.visa_info}
    itinerary_draft : {self.itinerary_draft}
    personalized_itinerary : {self.personalized_itinerary}
    accommodation : {self.accommodation}
    duration : {self.duration}
    start_date : {self.start_date}
    trip_tips : {self.trip_tips}
    destination_activity : {self.destination_activity}
    transportation : {self.transportation}
    next_node : {self.next_node}
    agent_input : {self.agent_input}
    """
    
    def __str__(self):
        return f"""
    chat_history : {self.chat_history}
    user_preferences : {self.user_preferences}
    current_location : {self.current_location}
    budget : {self.budget}
    destination : {self.destination}
    visa_info : {self.visa_info}
    itinerary_draft : {self.itinerary_draft}
    personalized_itinerary : {self.personalized_itinerary}
    accommodation : {self.accommodation}
    duration : {self.duration}
    start_date : {self.start_date}
    trip_tips : {self.trip_tips}
    destination_activity : {self.destination_activity}
    transportation : {self.transportation}
    next_node : {self.next_node}
    agent_input : {self.agent_input}
    """