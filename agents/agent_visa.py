import os
from dotenv.main import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .utils_agent import extract_json_from_response


load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# Should have its own llm instance 
llm_visa = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)




# Node as a function that return the task response (detinstiaon information in this example)
def get_location_visa(agent_input: dict):
    """
    Checks visa requirements based on nationality and destination.
    agent input should be valid json with `origin` and `destination` keys. if there is additional information, "other" key can be provided too.
    """
    
    VISA_SYSTEM_PROMPT = """
You are a helpful travel visa assistant.
Your task is to provide concise visa eligibility information for tourist travel only.

Given an origin country and a destination country, return the one or two easiest ways the traveler can be eligible to enter the destination.

If no visa is required, clearly state that and mention the allowed duration of stay.

If visa on arrival or eVisa is available, mention that, including how early to apply (if needed).

Include brief and practical tips, such as passport validity requirements, proof of return ticket, or travel insurance.

Respond in a clear and concise format (2–5 sentences max).
Focus only on tourism entry, not work, study, or immigration.
"""

    VISA_HUMAN_PROMPT = """
I’m planning a tourist trip from {origin} to {destination}.
Can you tell me the easiest one or two ways I can be eligible to enter the destination country?

Please include:

Whether I need a visa or not

If an eVisa or visa on arrival is available

How early I should apply (if applicable)

Any brief, useful tips like passport validity or return ticket requirements

Please keep it short and clear. Thanks!
"""
    
    if "other" in agent_input:
        VISA_HUMAN_PROMPT = VISA_HUMAN_PROMPT + f"\nOther information: {agent_input['other']} "
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", VISA_SYSTEM_PROMPT),
        ("human", VISA_HUMAN_PROMPT),
    ])
    chain = prompt | llm_visa
    result = chain.invoke({"origin": agent_input["origin"], "destination": agent_input["destination"]})
    return result.content

if __name__ == "__main__":
    response = get_location_visa("{'origin' : 'London' , 'destination' : 'Paris', 'other' : 'nationality is Iranian'}")
    print(response)