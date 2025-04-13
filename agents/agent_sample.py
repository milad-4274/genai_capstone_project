import os
from dotenv.main import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# Should have its own llm instance 
llm_destination = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY, temperature=1)


# Node as a function that return the task response (detinstiaon information in this example)
def research_destination(state):
    """
    Researches a destination and provides information. destination must be set before calling this agent.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Research '{destination}' and provide key information about attractions, local culture, and things to do."),
        ("human", "Tell me about {destination}."),
    ])
    chain = prompt | llm_destination
    result = chain.invoke({"destination": state.destination})
    return {"destination_info": result.content}