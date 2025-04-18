import os
import re
import requests
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from utils_agent import extract_json_from_response
from tools import get_weather
from data_models import DestinationRecommendationList
from PIL import Image
import io
from dateutil import parser as date_parser
from datetime import datetime
from dotenv.main import load_dotenv

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Setup Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Use Gemini multimodal
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1,
)

parser = PydanticOutputParser(pydantic_object=DestinationRecommendationList)

def download_image_bytes(image_url: str) -> Image.Image:
    """Download image from URL and return raw bytes."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


client = genai.Client(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-2.0-flash-exp")

def image_understanding(image_bytes:Image.Image) -> str:

    try:

        prompt = (
            f"Describe location and climate of the uploaded image similar to the provided examples.\n"
            f"1. What kind of location or activity this seems to be (e.g., ski resort, beach, hiking trail).\n"
            f"2. What kind of weather is represented.\n"
            f"3. Write in a friendly tone."
            f"\n\nExamples:\n"
        )

        EXAMPLES = [
            {
                "img": download_image_bytes("https://upload.wikimedia.org/wikipedia/commons/9/9d/Seychelles_Beach.jpg"),
                "answer": (
                    "1. A tranquil tropical beach perfect for sun‑lounging and gentle swimming.\n"
                    "2. Clear skies, bright sun, 28 °C with a light sea breeze.\n"
                    "3. Try Bora Bora or the Maldives in July.\n"
                    "4. Expect 27‑30 °C; plan for snorkeling, paddle‑boarding, and sunset cruises.\n"
                    "5. Friendly tone 🙂: *Pack reef‑safe sunscreen and a good book; paradise awaits!*"
                ),
            },
            {
                "img": download_image_bytes("https://upload.wikimedia.org/wikipedia/commons/0/03/Panorama_vom_Gornergrat-Zermatt.jpg"),
                "answer": (
                    "1. A high‑alpine ski resort with well‑groomed downhill runs.\n"
                    "2. Crisp winter weather: −5 °C, light powder snow, low humidity.\n"
                    "3. Consider Whistler (Canada) or St. Anton (Austria) in January.\n"
                    "4. Temps −10 °C to −2 °C; carve fresh pistes, try night skiing, warm up with après‑ski fondue.\n"
                    "5. Adventurous tone 🏂: *Strap in, breathe the icy air, and chase that first‑tracks adrenaline!*"
                ),
            },
            ]
        # Build the content list:  (ex1_img, ex1_answer, ex2_img, ex2_answer, target_img, prompt)
        content = []
        for ex in EXAMPLES:
            content.extend([ex["img"], ex["answer"]])
        content.extend([image_bytes, prompt])

        # response = model.generate_content(content)
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=content
        )

        return response.text.strip()

    except Exception as e:
        return f"Error: {e}"


def clean_preferences_and_extract_image(preferences: str):
    pattern = r'(https?://[^\s]+\.(?:jpg|jpeg|png|webp|gif))'
    match = re.search(pattern, preferences, re.IGNORECASE)
    image_url = match.group(1) if match else None
    cleaned = re.sub(pattern, '', preferences).strip()
    return cleaned, image_url

def destination_recommender(agent_input: dict) -> dict:
    """
    Uses Gemini 2.0-flash=exp to interpret the climate/activity from an image (local or URL),
    then recommends similar destinations for that type of environment in a specific month.
    """

    required_fields = ["travel_date", "duration", "budget", "accommodation_preferences"]
    for f in required_fields:
        if f not in agent_input:
            return f" Missing required field: {f}"
        
    style = agent_input.get("style", "friendly")

     # Try to extract image from preferences if not already present
    if "image_url" not in agent_input or not agent_input["image_url"]:
        cleaned_prefs, image_url = clean_preferences_and_extract_image(agent_input["accommodation_preferences"])
        if image_url:
            agent_input["image_url"] = image_url
            agent_input["accommodation_preferences"] = cleaned_prefs
    try:
        image_bytes = download_image_bytes(agent_input["image_url"])
        img_content = image_understanding(image_bytes)

        # Parse flexible travel date
        try:
            travel_start_date = date_parser.parse(agent_input["travel_date"], fuzzy=True, default=datetime.now())
        except Exception as e:
            return f"Could not parse travel date: {e}"

        today = datetime.today()
        days_until_trip = (travel_start_date - today).days

        # Build the prompt
        prompt = (
            f"The user is planning a trip starting on {travel_start_date.strftime('%B %d, %Y')}, lasting {agent_input['duration']}, "
            f"with a total budget of {agent_input['budget']}.\n\n"
            f"The user's preference for climate or activity is based on "
            f"{'the uploaded image, which suggests: ' + img_content if img_content else 'their written input: ' + agent_input['accommodation_preferences']}.\n\n"
            f"Recommend **3 travel destinations cities** that:\n"
            f"- Match the user's interests and climate\n"
            f"- Fit within the time and budget\n"
            f"- Are realistic to visit during that season\n\n"
            f"For each destination, provide:\n"
            f"1. Name of the city and its country\n"
            f"2. Description of activities and vibe\n"
            f"3. Suggested experience\n"
            f"4. Why it's a good match\n\n"
            f"Make sure that you Do NOT provide any information on the tempreture of those cities you recommend."
            f"Return the result in this JSON format:\n{parser.get_format_instructions()}\n"
            f"Respond in a {style} tone."
        )


        temp_response = llm.invoke(prompt)
        cities = [item.location_names[0] for item in parser.parse(temp_response.content).root]
        # Build weather info string
        weather_infos = []  # List of weather summaries (one per city)

        for city in cities:
            if days_until_trip <= 14:
                weather_data = get_weather(city, travel_start_date)
                if weather_data:
                    min_c = weather_data.get("min_temp")
                    max_c = weather_data.get("max_temp")
                    summary = weather_data.get("weather_summary", "pleasant weather")
                    weather_summary = (
                        f"{city} on {travel_start_date}: {summary} with highs around {max_c}°C and lows around {min_c}°C."
                    )
                else:
                    weather_summary = (
                        f"{city} on {travel_start_date}: Forecast unavailable. Please check local weather updates closer to your trip."
                    )
            else:
                weather_summary = (
                    f"{city} on {travel_start_date}: Estimate the likely weather based on historical climate for that time of year."
                )

            weather_infos.append(weather_summary)

        # Join them into a single prompt
        joined_weather = "\n".join(weather_infos)

        final_prompt = (
            f"🌍 Trip Summary Request 🌞\n\n"
            f"📅 Trip starts on: {travel_start_date.strftime('%B %d, %Y')}\n"
            f"🗺️ Recommended Destinations:\n\n{temp_response.content}\n\n"
            f"🌤️ Weather Forecast specific for the duration of your stay:\n\n{joined_weather}\n\n"
            "✈️ Now, write a combined travel-friendly summary incorporating both destination descriptions and their weather conditions.\n"
            f"Return the result in this JSON format:\n{parser.get_format_instructions()}\n"
            f"Respond in a {style} tone."
        )

        response = llm.invoke(final_prompt)
        return  parser.parse(response.content).model_dump()

    except Exception as e:
        return f"❌ Error: {e}"
    
if __name__ == "__main__":
    response = {
        "travel_date": "April 22th",
        "duration": "5 days",
        "budget": "$1500",
        "accommodation_preferences": "Let's go somewhere like this! Here's what I mean: https://upload.wikimedia.org/wikipedia/commons/f/f9/Playa_de_El_Buzo_2_de_mayo_de_2009.jpg",
    }

    # the supervisor has to make sure if it calls this function, 
    # only when we know that the destination value is empty
    print(destination_recommender(response))