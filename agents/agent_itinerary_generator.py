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
llm_itinerary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)




# Node as a function that return the task response (detinstiaon information in this example)
def generate_itinerary(agent_input : str):
    """
    generate trip itinerary 
    """
    
    ITINERARY_SYSTEM_PROMPT = """
You are an expert travel assistant that builds custom tourist itineraries.

Your goal is to create a structured itinerary based on:
- the user's travel preferences,
- destination,
- trip duration,
- travel date,
- and provide destination-specific cultural or practical tips.

Only suggest realistic, enjoyable, and culturally respectful tourist activities. Include food, sightseeing, walking tours, or nature experiences when appropriate.

Respond in the following JSON format:
{format_instructions}
""".strip()

    ITINERARY_HUMAN_PROMPT = """
I'm planning a trip with the following details:

- Destination: {destination}
- Travel date: {travel_date}
- Duration: {duration} days
- My preferences: {preferences}
- Budget: {budget}

Please create a day-by-day itinerary with tips and suggested activities. Make sure to include general tips and any assumptions you made.
""".strip()
    
    try:
        input_dict = extract_json_from_response(agent_input)
    except Exception as e:
        return "Agent Input Error: invalid json as input. call the agent again with valid input", repr(e)
    
    required_input_features = ["destination", "travel_date", "duration", "preferences", "budget"]
    
    for feature in required_input_features:
        if feature not in input_dict:
            return "Agent Input Error: the input must include {feature}."
    
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ITINERARY_SYSTEM_PROMPT),
        ("human", ITINERARY_HUMAN_PROMPT),
    ])
    
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        **input_dict
    )
    response = llm_itinerary.invoke(formatted_prompt)
    result = parser.parse(response.content)
    return result.model_dump()

if __name__ == "__main__":
    response = generate_itinerary('''{
  "destination": "London",
  "travel_date": "April 23th",
  "duration": "5 Days",
  "preferences": "I love Arsenal, I enjoy going and watching historic sites I love doing physical activity I love to see local culture I love to see some of Iranian parts of cities or stores",
  "budget" : "1500 $",
}''')
    print(response)