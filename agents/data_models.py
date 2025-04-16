
from pydantic import BaseModel, Field, RootModel
from typing import List, Optional


# Define the structured output schema
class Activity(BaseModel):
    title: str
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


class SillyTravelBriefing(BaseModel):
    destination: str 
    weather: str 
    outfit: str 
    language_tip: str 
    safety_tip: str 
    closing_remark: str 

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
