from google import genai
from google.genai import types
import os
from dotenv.main import load_dotenv
from IPython.display import Markdown, HTML, display


# Define a retry policy. The model might make multiple consecutive calls automatically
# for a complex query, this ensures the client retries if it hits quota limits.
from google.api_core import retry
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


client = genai.Client(api_key=GOOGLE_API_KEY)

is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})

if not hasattr(genai.models.Models.generate_content, '__wrapped__'):
  genai.models.Models.generate_content = retry.Retry(
      predicate=is_retriable)(genai.models.Models.generate_content)
  
config_with_search = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

def query_with_grounding():
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        # contents="What is the cheapest flight from LA to BOSTON at 27 april 2025? available to buy and preferably in the morning, select a flight only provide the final link to the selected flight",
        contents="What is the cheapest accomodation in Boston for two people? at 27 april 2025. provide the link to the selected hotel",
        config=config_with_search,
    )
    return response.candidates[0]

rc = query_with_grounding()
# Check if there were any tool calls
if rc.tool_calls:
    print("Tool Calls:")
    for tool_call in rc.tool_calls:
        if tool_call.function.name == "google_search":
            arguments = tool_call.function.arguments
            print(f"  Function: {tool_call.function.name}")
            print(f"  Arguments: {arguments}")

            # The search results are in the 'parts' of the 'tool_outputs'
            if rc.prompt_feedback and rc.prompt_feedback.tool_outputs:
                for tool_output in rc.prompt_feedback.tool_outputs:
                    if tool_output.tool_call_id == tool_call.id:
                        print("  Search Results:")
                        search_results = tool_output.output.parts[0].text
                        print(search_results)
                        # You can further process 'search_results' to find the desired link
                        # This will likely be a string containing multiple search snippets.
                        # You'll need to parse this string to extract URLs.
else:
    print("No tool calls were made.")
