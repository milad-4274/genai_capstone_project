import os
from dotenv.main import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from utils_agent import extract_json_from_response
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from data_models import Activity, DailyActivity, Itinerary


load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# Create parser
parser = PydanticOutputParser(pydantic_object=Itinerary)

# Should have its own llm instance 
llm_itinerary_reviewer = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)




# Node as a function that return the task response (detinstiaon information in this example)
def review_itinerary(agent_input : str):
    """
    generate trip itinerary 
    """
    
    REVIEW_ITINERARY_SYSTEM_PROMPT = """
You are an expert travel assistant that reviews tourist itineraries.

Your goal is to modify the given itinerary based on:
- the user's travel preferences,
- trip duration,
- travel date,

The given itinerary should be fine, try to make it more personalized, tune activities based and replace them if they're not related. You can add more tips based on activities too.

The initial itinerary is in json format and respond in the following JSON format:
{format_instructions}
""".strip()

    REVIEW_ITINERARY_HUMAN_PROMPT = """
It is my itinerary to go to {destination} in {travel_date} for {duration}. the itinerary is generated based on the following preferences:
** User Preferences **: 
{preferences}
Budget: {budget}

** Itinerary **:
{itinerary}

make it more personalized, tune activities based and replace them if they're not related
""".strip()
    
    try:
        input_dict = extract_json_from_response(agent_input)
    except Exception as e:
        return "Agent Input Error: invalid json as input. call the agent again with valid input", repr(e)
    
    required_input_features = ["destination", "travel_date", "duration", "preferences", "budget", "itinerary"]
    
    for feature in required_input_features:
        if feature not in input_dict:
            return f"Agent Input Error: the input must include {feature}."
    
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEW_ITINERARY_SYSTEM_PROMPT),
        ("human", REVIEW_ITINERARY_HUMAN_PROMPT),
    ])
    
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        **input_dict
    )
    response = llm_itinerary_reviewer.invoke(formatted_prompt)
    result = parser.parse(response.content)
    return result.model_dump()

if __name__ == "__main__":
    import json
    destination = "London"
    travel_date = "April 23th"
    duration = "5 Days"
    preferences = "I love Arsenal, I enjoy going and watching historic sites I love doing physical activity I love to see local culture I love to see some of Iranian parts of cities or stores"
    budget = "1500 $"
    itinierary = {"assumptions": ["Budget is per person.", "You are arriving in London on April 23rd and departing on April 27th.", "You are interested in a mix of historical sites, cultural experiences, and physical activities.", "You are able to walk moderate distances."], "general_tips": ["Oyster card or contactless payment is the easiest way to get around on public transport.", "Book accommodations and tours in advance, especially during peak season.", "Wear comfortable shoes, as you'll be doing a lot of walking.", "Tipping is not mandatory in the UK, but it is appreciated for good service (10-15%).", "Be aware of your surroundings and keep your belongings safe, especially in crowded areas."], "daily_activities": [{"day": 1, "title": "Arrival & Arsenal Immersion", "daily_tips": "Check the Arsenal match schedule in advance. Consider a stadium tour even if there isn't a match.", "activities": [{"title": "Arrive at Airport & Transfer to Accommodation", "activity_type": "Transportation/Check-in", "time_of_day": "Morning", "estimated_start_time": "9:00 AM", "estimated_duration": "2 hours", "estimated_budget": "$50"}, {"title": "Arsenal Stadium Tour", "activity_type": "Sports/Tour", "time_of_day": "Afternoon", "estimated_start_time": "12:00 PM", "estimated_duration": "3 hours", "estimated_budget": "$40"}, {"title": "Emirates Stadium Visit & Merchandise Shopping", "activity_type": "Shopping/Leisure", "time_of_day": "Afternoon", "estimated_start_time": "3:00 PM", "estimated_duration": "2 hours", "estimated_budget": "$50"}, {"title": "Dinner near Emirates Stadium", "activity_type": "Food", "time_of_day": "Evening", "estimated_start_time": "6:00 PM", "estimated_duration": "1.5 hours", "estimated_budget": "$30"}]}, {"day": 2, "title": "Historical London & Iranian Influence", "daily_tips": "Wear comfortable shoes for walking. Brick Lane is best explored during the day.", "activities": [{"title": "Tower of London & Tower Bridge", "activity_type": "Historical Site/Tour", "time_of_day": "Morning", "estimated_start_time": "9:00 AM", "estimated_duration": "4 hours", "estimated_budget": "$40"}, {"title": "Brick Lane Exploration", "activity_type": "Cultural/Food", "time_of_day": "Afternoon", "estimated_start_time": "1:00 PM", "estimated_duration": "3 hours", "estimated_budget": "$30"}, {"title": "Iranian Restaurant in Edgware", "activity_type": "Food", "time_of_day": "Evening", "estimated_start_time": "6:00 PM", "estimated_duration": "2 hours", "estimated_budget": "$40"}]}, {"day": 3, "title": "Royal London & Parks", "daily_tips": "Check the schedule for Changing of the Guard in advance. Pack a picnic for the park.", "activities": [{"title": "Buckingham Palace & Changing of the Guard", "activity_type": "Historical Site/Ceremony", "time_of_day": "Morning", "estimated_start_time": "10:00 AM", "estimated_duration": "2 hours", "estimated_budget": "$0"}, {"title": "St. James's Park & Green Park Walk", "activity_type": "Nature/Walking", "time_of_day": "Afternoon", "estimated_start_time": "12:00 PM", "estimated_duration": "3 hours", "estimated_budget": "$0"}, {"title": "Westminster Abbey", "activity_type": "Historical Site/Tour", "time_of_day": "Afternoon", "estimated_start_time": "3:00 PM", "estimated_duration": "2 hours", "estimated_budget": "$30"}, {"title": "Dinner in Westminster", "activity_type": "Food", "time_of_day": "Evening", "estimated_start_time": "6:00 PM", "estimated_duration": "1.5 hours", "estimated_budget": "$30"}]}, {"day": 4, "title": "Markets, Museums & Physical Challenge", "daily_tips": "Borough Market can get very crowded on weekends. Choose a walking tour based on your fitness level.", "activities": [{"title": "Borough Market", "activity_type": "Food/Market", "time_of_day": "Morning", "estimated_start_time": "10:00 AM", "estimated_duration": "2 hours", "estimated_budget": "$30"}, {"title": "Tate Modern", "activity_type": "Museum/Art", "time_of_day": "Afternoon", "estimated_start_time": "12:00 PM", "estimated_duration": "3 hours", "estimated_budget": "$0"}, {"title": "Thames Path Walk", "activity_type": "Physical Activity/Nature", "time_of_day": "Afternoon", "estimated_start_time": "3:00 PM", "estimated_duration": "3 hours", "estimated_budget": "$0"}, {"title": "Dinner near South Bank", "activity_type": "Food", "time_of_day": "Evening", "estimated_start_time": "6:00 PM", "estimated_duration": "1.5 hours", "estimated_budget": "$30"}]}, {"day": 5, "title": "Departure", "daily_tips": "Allow ample time for travel to the airport, especially during rush hour.", "activities": [{"title": "Free Morning - Shopping or revisit", "activity_type": "Leisure/Shopping", "time_of_day": "Morning", "estimated_start_time": "9:00 AM", "estimated_duration": "3 hours", "estimated_budget": "$50"}, {"title": "Transfer to Airport", "activity_type": "Transportation", "time_of_day": "Afternoon", "estimated_start_time": "12:00 PM", "estimated_duration": "2 hours", "estimated_budget": "$50"}]}]}
    agent_input = f"""{{
    "destination": "{destination}",
    "travel_date": "{travel_date}",
    "duration": "{duration}",
    "preferences": "{preferences}",
    "budget": "{budget}",
    "itinerary": {json.dumps(itinierary)}
}}
"""

    print(agent_input)
    
    response = review_itinerary(agent_input=agent_input)
    print(json.dumps(response))
    