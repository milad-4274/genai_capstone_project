import os
from dotenv.main import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from agents.utils_agent import extract_json_from_response
from google.genai import types
from google import genai
from google.api_core import retry

from agents.data_models import Transportation, TransportationList
from langchain.output_parsers import PydanticOutputParser
from langsmith import traceable, Client
from typing import List

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

  

# Should have its own llm instance 
# llm_transportation = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

client = Client()


if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)

config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)


# Node as a function that return the task response (detinstiaon information in this example)
@traceable(name="gemini_grounding_transportation")
def get_transportation(agent_input : str):
    """
    Recommend the transportation for the trip based on inputs
    agent input should be valid json with `origin`, `destination`, `transportation_preferences`, `start_date` and `duration` keys.
    """
    
    # Create parser
    parser = PydanticOutputParser(pydantic_object=TransportationList)
    
    TRANSPORTATION_SYSTEM_PROMPT = """
You are a helpful transportation expert.
Your task is to recommend best option(s) to travel between origin and destination.

Given an origin country and a destination country, return the one or more easiest ways the traveler can go from origin to destination.

you can search for all good options (e.g flight, train, bus, etc) for the start and end date to estimate the cost of transportation. for example you can search for round trip / or one way with flight one with train / flight in different costs / flight in different times for better timing and etc.

Respond in the following JSON format:
{format_instructions}
""".strip()

    TRANSPORTATION_HUMAN_PROMPT = """
I'm planning a tourist trip from {origin} to {destination} on {start_date} for {duration}. 
here are my preferences about transportation:
{transportation_preferences}
please find the best ways for me.
""".strip()
    
    try:
        input_dict = extract_json_from_response(agent_input)
    except:
        return "invalid json as input. call the agent again with valid input"
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRANSPORTATION_SYSTEM_PROMPT),
        ("human", TRANSPORTATION_HUMAN_PROMPT),
    ])
    
    formatted_prompt = prompt.format_messages(
        format_instructions=parser.get_format_instructions(),
        **input_dict
    )
    
    # Convert the formatted prompt to a string
    prompt_text = "\n".join([msg.content for msg in formatted_prompt])
    
    response = genai_client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt_text,
        config=config_with_search,
    )
    
    result = parser.parse(response.text)
    return result.model_dump()

if __name__ == "__main__":
    response = get_transportation("{'origin' : 'London' , 'destination' : 'Paris', 'transportation_preferences' : 'I love traveling by ship', 'start_date': 'April 23th', 'duration' : '5 days'}")
    print(response)