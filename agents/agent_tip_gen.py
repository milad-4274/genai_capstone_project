import os
import re
import json
from typing import List, Dict, Optional 
from dotenv import load_dotenv
from .tools import get_weather
from .data_models import SillyTravelBriefing, TravelPreferences
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from datetime import datetime, timedelta 
from langchain_community.utilities import GoogleSearchAPIWrapper
from dateutil import parser as date_parser
from langchain.output_parsers import PydanticOutputParser
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)
Google_Search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY)

# Create parser
parser_brief = PydanticOutputParser(pydantic_object=SillyTravelBriefing)

def parse_duration(duration_str: str) -> int:
    """Parses duration strings like '5 days', '1 week' into number of days."""
    duration_str = duration_str.lower().strip()
    days = 1 # Default to 1 day if parsing fails

    # Regex to find numbers followed by day/week/month variations
    match = re.match(r"(\d+)\s*(day|week|month)s?", duration_str)
    if match:
        num = int(match.group(1))
        unit = match.group(2)
        if unit == "day":
            days = num
        elif unit == "week":
            days = num * 7
        elif unit == "month":
            # Approximate month duration, adjust if needed
            days = num * 30
    elif "weekend" in duration_str:
        days = 2 # Or 3, depending on definition
    else:
        # Try converting directly if it's just a number
        days = int(duration_str)

    # Ensure minimum of 1 day and reasonable maximum (e.g., 16 for many free forecast APIs)
    days = max(1, min(days, 16))
    return days

def get_daily_forecast(destination: str, start_date: datetime, num_days: int) -> Optional[List[dict]]:
    """
    Fetches daily forecast data (min/max temp, description) for `num_days`.
    Replace `get_weather()` with your actual API logic.
    """
    try:
        forecast_data = []
        for i in range(num_days):
            current_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
            weather_data = get_weather(destination, current_date)
            
            if isinstance(weather_data, dict) and all(k in weather_data for k in ["date", "min_temp", "max_temp", "weather_summary"]):
                forecast_data.append({
                    'date': weather_data["date"],
                    'min_temp': weather_data["min_temp"],
                    'max_temp': weather_data["max_temp"],
                    'description': weather_data["weather_summary"]
                })
            else:
                print(f"⚠️ Skipping {current_date}: Incomplete or invalid data → {weather_data}")
                continue

        return forecast_data
    
    except Exception as e:
        print(f"Error in get_daily_forecast: {e}")
        return None


def get_clothing_tip_for_range(min_temp, max_temp, description: str) -> str:
    """Generates clothing advice based on min/max temperature range and description."""
    tip = ""
    has_min = isinstance(min_temp, (int, float))
    has_max = isinstance(max_temp, (int, float))
    has_temps = has_min and has_max
    rain_possible = "rain" in description.lower() or "shower" in description.lower()

    if not has_temps:
        tip = "Temperature data unavailable. Pack layers and hope for the best!"
    elif max_temp < 5:
        tip = f"Very Cold (Low {min_temp}°C, High {max_temp}°C)! Bundle up: Heavy coat, thermal layers, hat, gloves, scarf are essential."
    elif max_temp < 12:
        tip = f"Cold (Low {min_temp}°C, High {max_temp}°C). Wear layers: Warm jacket or coat, sweater/fleece, long pants. Consider hat/gloves, especially if windy or near {min_temp}°C."
    elif max_temp < 18:
        tip = f"Cool (Low {min_temp}°C, High {max_temp}°C). A medium jacket or layered sweaters recommended. Long pants ideal. Layers help adjust from cooler {min_temp}°C periods."
    elif max_temp < 25:
        tip = f"Mild/Warm (Low {min_temp}°C, High {max_temp}°C). Variable! T-shirt with a light jacket/sweater for cooler {min_temp}°C times (mornings/evenings). Pants or shorts depending on the {max_temp}°C."
    elif max_temp < 30:
        tip = f"Warm (Low {min_temp}°C, High {max_temp}°C). T-shirts, light pants or shorts. A very light layer might be nice for evenings near {min_temp}°C."
    else: # max_temp >= 30
        tip = f"Hot (Low {min_temp}°C, High {max_temp}°C)! Pack light, breathable clothing (cotton, linen), shorts, tank tops. Stay hydrated and use sun protection."

    if has_temps and abs(max_temp - min_temp) >= 10:
         tip += " Significant temperature swing: Layers are key!"

    if rain_possible:
        tip += " Pack a raincoat or umbrella!"

    return tip
def get_multi_day_weather_and_clothing_context(destination: str, start_date_str: str, duration_str: str) -> str:
    """
    Fetches multi-day forecast and generates weather summary and clothing tips
    for the duration of the stay.
    """
    # Parse Start Date
    if isinstance(start_date_str, datetime):
        start_date_obj = start_date_str
    else:
        start_date_obj = date_parser.parse(start_date_str)
    
    # Parse Duration
    num_days = parse_duration(duration_str)

    # Get Forecast Data
    daily_forecasts = get_daily_forecast(destination, start_date_obj, num_days)

    # Process Forecast and Generate Context String
    if not daily_forecasts:
        # get_daily_forecast already logged the error
        return f"Could not retrieve a reliable weather forecast for {destination} for {num_days} days starting {start_date_obj.strftime('%b %d')}. Please check the location or try again later."

    full_context = f"Weather & Clothing Tips for {destination} ({num_days} days):\n"
    if len(daily_forecasts) < num_days:
        full_context += f"(Note: Forecast available for only {len(daily_forecasts)} days)\n"


    for day_forecast in daily_forecasts:
        date = day_forecast['date']
        min_t = day_forecast['min_temp']
        max_t = day_forecast['max_temp']
        desc = day_forecast['description']

        # Format date nicely (e.g., Apr 23rd)
        date_obj = date_parser.parse(date)
        date_formatted = date_obj.strftime('%b %d')

        # Format temps, handling N/A
        temp_str = f"High: {max_t}°C, Low: {min_t}°C"
        if min_t == 'N/A' or max_t == 'N/A':
            temp_str = "Temperature unavailable"


        weather_summary = f"- {date_formatted}: {desc}. {temp_str}."
        clothing_tip = get_clothing_tip_for_range(min_t, max_t, desc)

        full_context += f"{weather_summary}\n  Clothing: {clothing_tip}\n"

    return full_context.strip()

def get_tool_search_results(query: str, tool_name: str) -> str:
    """Helper function to run Google Search and handle results/errors."""
    try:
        results = Google_Search.run(query)
        if results and results != "No good Google Search Result was found":
            # Limit length to avoid overwhelming the LLM context
            return f"{tool_name} context: {results[:600]}..."
        else:
            return f"Couldn't find specific {tool_name} info via search."
    except Exception as e:
        return f"Error searching for {tool_name} information."

def get_cultural_context(destination: str, preferences: str) -> str:
    """Gets cultural context using Google Search."""
    query = f"Unique funny cultural quirks or tips for tourists in {destination} related to {preferences}"
    return get_tool_search_results(query, "Cultural tips")

def get_language_context(destination: str) -> str:
    """Gets language context using Google Search."""
    # Focused query for fun/essential phrases
    query = f"Funny or essential basic local language phrases for tourists in {destination}"
    return get_tool_search_results(query, "Language tips")

def get_safety_context(destination: str) -> str:
    """Gets safety context using Google Search."""
    query = f"Key safety tips for tourists in {destination} (keep it concise)"
    return get_tool_search_results(query, "Safety tips")


def silly_travel_stylist_structured(agent_input: dict) -> Dict:
    """
    Takes travel preferences as JSON, calls tools to gather context,
    and uses an LLM to generate a structured SillyTravelBriefing.
    """
    try:
        prefs = TravelPreferences(**agent_input)
    except json.JSONDecodeError as e:
        print(agent_input)
        return {"error": f"Invalid JSON input. Please provide valid JSON. Error: {e}"}
    except Exception as e: # Catches Pydantic validation errors too
        print(agent_input)
        return {"error": f"Could not parse input data. Check fields. Error: {e}"}

    if isinstance(prefs.travel_date, str):
        from dateutil import parser
        start_date = parser.parse(prefs.travel_date)
    else:
        start_date = prefs.travel_date

    duration_days = int(re.search(r'\d+', agent_input["duration"]).group())
    end_date = start_date + timedelta(days=duration_days)
    ten_days_from_now = datetime.now() + timedelta(days=10)

    # 1. Gather Context from Tools
    if end_date <= ten_days_from_now:
        weather_context = get_multi_day_weather_and_clothing_context(
            agent_input["destination"],
            start_date,
            agent_input["duration"]
        )
    else:
        prompt = (
        f"I'm planning a trip to {agent_input['destination']} in {agent_input['travel_date']}."
        f"Can you give me tips on what kind of clothes to pack based on the typical weather?"
        )
        weather_context = google_llm.invoke(prompt).content.strip()
    cultural_context = get_cultural_context(agent_input["destination"], agent_input["preferences"])
    language_context = get_language_context(agent_input["destination"])
    safety_context = get_safety_context(agent_input["destination"])

    # 2. Prepare Input for LLM
    combined_context = f"""
    Travel Destination: {agent_input["destination"]}
    Travel Date: {agent_input["travel_date"]}
    Trip Duration: {agent_input["duration"]}
    Traveler Preferences: {agent_input["preferences"]}
    Budget: {agent_input["budget"] or 'Not specified'}

    Gathered Information:
    Weather/Clothing Context: {weather_context}
    Cultural Context: {cultural_context}
    Language Context: {language_context}
    Safety Context: {safety_context}
    """

    # 3. Define Prompt for Structured Output Generation
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         "You are the 'Silly Travel Stylist'. Your task is to create a humorous and quirky, yet informative, travel briefing based on the provided context. "
         "Use the gathered information about weather, culture, language, and safety, along with the traveler's preferences. "
         "Inject humor and personality into each field. "
         "Generate the output STRICTLY in the requested JSON format matching the SillyTravelBriefing schema. "
         "Focus on being funny but still providing a hint of useful advice within the humor. "
         "Refer to the traveler's preferences when suggesting activities."),
        ("human",
         "Please generate a SillyTravelBriefing JSON for the following travel plan:\n\n{context}")
    ])

    # 4. Create LLM Chain with Structured Output
    # This ensures the LLM output conforms to the Pydantic model
    structured_llm = google_llm.with_structured_output(SillyTravelBriefing)
    chain = prompt_template | structured_llm

    # 5. Invoke Chain and Get Structured Output
    try:
        response = chain.invoke({"context": combined_context})
        return str(response.model_dump())

    except Exception as e:
        # Fallback or error reporting
        error_message = f"Failed to generate the silly travel briefing. The LLM might be having an off day! Error: {e}"
        return {"error": error_message}



if __name__ == "__main__":
    travel_request_json = '''{
      "destination": "London",
      "travel_date": "April 18th 2025",
      "duration": "5 Days",
      "preferences": "I love Arsenal, I enjoy going and watching historic sites I love doing physical activity I love to see local culture I love to see some of Iranian parts of cities or stores",
      "budget" : "1500 $"
    }'''
    briefing_output = silly_travel_stylist_structured(travel_request_json)

    print(json.dumps(briefing_output.dict(), indent=2))

    