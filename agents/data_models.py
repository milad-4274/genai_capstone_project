
from pydantic import BaseModel, Field
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