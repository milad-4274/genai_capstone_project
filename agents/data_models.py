
from pydantic import BaseModel, Field, RootModel
from typing import List, Optional
from  pydantic import BaseModel as PydanticBaseModel

# Define the structured output schema
class Activity(BaseModel):
    title: str
    description: str
    activity_type: str
    time_of_day: str
    estimated_start_time: str
    estimated_duration: str
    estimated_budget: str

class DailyActivity(BaseModel):
    day: int
    title: str
    daily_tips: str
    activities: List[Activity]

class Itinerary(BaseModel):
    assumptions: List[str]
    general_tips: List[str]
    daily_activities: List[DailyActivity]



class Activity(BaseModel):
    title: str
    activity_type: str
    time_of_day: str
    estimated_start_time: str
    estimated_duration: str
    estimated_budget: str
    
class Transportation(BaseModel):
    mode: str
    description: str
    cost: str 

class TransportationList(BaseModel):
    items: List[Transportation]

class SillyTravelBriefing(PydanticBaseModel):
    """Structured output for the silly travel briefing."""
    destination: str 
    weather_summary: str 
    clothing_tip: str 
    cultural_tip: str 
    language_tip: str 
    safety_tip: str 
    activity_suggestion: str 
    closing_remark: str 

class TravelPreferences(PydanticBaseModel):
    """Input model for travel details."""
    destination: str 
    travel_date: str 
    duration: str 
    preferences: str 
    budget: Optional[str] 
class DestinationRecommendation(BaseModel):
    location_type: str
    location_names: list[str]
    weather_description: str
    similar_destinations: List[str]
    expected_temperatures: str
    recommended_activities: List[str]
    commentary: str
class DestinationRecommendationList(RootModel[List[DestinationRecommendation]]):
    pass
