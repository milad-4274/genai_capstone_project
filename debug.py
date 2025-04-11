import os
from dotenv.main import load_dotenv
from IPython.display import Image, display
from pprint import pprint


from typing import Annotated
from typing_extensions import TypedDict
from typing import Literal

from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages.ai import AIMessage
from langgraph.prebuilt import ToolNode



from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID= os.getenv("GOOGLE_CSE_ID")
GOOGLE_SEARCH_KEY= os.getenv("GOOGLE_SEARCH_KEY")

os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME")  # replace with your Kaggle username
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_KEY")          # replace with your Kaggle API key
class SearchState(TypedDict):
    """State representing the customer's search conversation."""

    # The chat conversation. This preserves the conversation history
    # between nodes. The `add_messages` annotation indicates to LangGraph
    # that state is updated by appending returned messages, not replacing
    # them.
    messages: Annotated[list, add_messages]

    # The customer's in-progress search.
    search: list[str]

    # Flag indicating that the search is placed and completed.
    finished: bool

TRIPADVISOR_SYSINT = (
    "system",  # 'system' indicates the message is a system instruction.
    "You are a helpful and friendly Trip Advisor bot. Come up with 5 destinations with given budget estimation "
    "based on the preferred activities or weather they wanna be at. A human will ask you for recommendations "
    "and information about transportation to use, places to visit, activities to do, restaurants to eat at, and "
    "accommodations to stay in. Your primary goal is to provide relevant and accurate information "
    "based on the user's requests, helping them plan their trip effectively. "
    "You can ask clarifying questions to better understand their preferences and needs. "
    "Avoid engaging in off-topic discussions and focus solely on trip-related inquiries. "
    "\n\n"
    "You have access to several tools to assist the user:"
    "\n"
    "- `activity_search`: Use this tool to find information about specific places, attractions, "
    "restaurants, or museums. The input should be a detailed query including location "
    "and keywords describing what the user is looking for (e.g., 'best beaches in Barcelona', "
    "'romantic restaurants near Sagrada Familia', 'budget-friendly hotels in the Gothic Quarter')."
    "If the user asks for a list of places, ALWAYS respond with a "
    "search_places tool call (no text) whose arguments include at least "
    '{"query": "<user request>"} . After the tool returns, summarise the results.'
    "\n"
    "- `accommodation_search`: - `accommodation_search`: Use this tool to search for hotel accommodations"
    " and reviews based on user preferences. The input should be a detailed query that includes location "
    "and relevant keywords describing what the user is looking for (e.g., affordable hotels near Central Park, "
    "luxury accommodations with a sea view in Miami, budget-friendly stays in the historic district" 
    "If the user asks for accommodation recommendations, ALWAYS respond with an `accommodation_search` tool call (no additional text) whose arguments include at least"
    '{"query": "<user request>".After the tool returns, summarise the results.'
    "When responding to the user, synthesize information from these tools into a coherent and "
    "helpful answer. If a user asks for recommendations, use `search_places` with relevant keywords "
    "and present a few options. You can then use `get_place_details` if the user expresses interest "
    "in a specific option. If the user asks for how to get somewhere, use `get_directions`. If there's "
    "a language barrier, offer to use `translate_text`."
    "\n\n"
    "Remember to be polite and helpful throughout the interaction. If you cannot find information "
    "related to the user's request, acknowledge this and suggest alternative ways you might be able to assist."
    "\n\n"
    "If any of the tools are unavailable, you can break the fourth wall and tell the user that "
    "they have not implemented them yet and should keep reading to do so.",
)

WELCOME_MSG = "Welcome to the TripoBot. Type `q` to quit. What is your dream trip? Where are you residing now? what activity do you like to do? Do you like blind trip program? Which weather do you prefer? "

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


def chatbot(state: SearchState) -> SearchState:
    """The chatbot itself. A simple wrapper around the model's own chat interface."""
    message_history = [TRIPADVISOR_SYSINT] + state["messages"]
    return {"messages": [llm.invoke(message_history)]}


def human_node(state: SearchState) -> SearchState:
    """Display the last model message to the user, and receive the user's input."""
    last_msg = state["messages"][-1]
    print("Model:", last_msg.content)

    user_input = input("User: ")

    # If it looks like the user is trying to quit, flag the conversation
    # as over.
    if user_input in {"q", "quit", "exit", "goodbye"}:
        state["finished"] = True

    return state | {"messages": [("user", user_input)]}

def maybe_exit_human_node(state: SearchState) -> Literal["chatbot", "__end__"]:
    """Route to the chatbot, unless it looks like the user is exiting."""
    if state.get("finished", False):
        return END
    else:
        return "chatbot"

# The default recursion limit for traversing nodes is 25 - setting it higher means
# you can try a more complex search with multiple steps and round-trips (and you
# can chat for longer!)
config = {"recursion_limit": 25}

from langchain_core.tools import tool
from typing import List, Dict, Optional
import requests
import json
import traceback

@tool("activity_search")
def activity_search(
    query: str,
    num_results: int = 5,      
    how_many: int = 5,            
    style: str = "friendly",    
    **kwargs,
) -> str:
    """
    Search Google (Programmable Search Engine) and return a **summarised list**
    of the best activities/attractions for the user based on a given location.

    Args:
        query: Natural‑language search, e.g. "best museums in Paris".
        num_results: Raw results fetched from Google (max 10).
        how_many: Items to keep after LLM re‑ranking.
        style: Output tone ("friendly", "formal", "bullet").
    """
    if not (GOOGLE_SEARCH_KEY and GOOGLE_CSE_ID):
        raise RuntimeError(
            "Set GOOGLE_SEARCH_API_KEY and GOOGLE_CSE_ID env vars first."
        )

    # Google Search API ───────────────────────────
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_SEARCH_KEY,
        "cx":  GOOGLE_CSE_ID,
        "q":   query,
        "num": min(num_results, 5),
    }

    try:
        raw = requests.get(url, params=params, timeout=10).json()
        items: List[Dict] = raw.get("items", [])
        # titles = [item.get("title") for item in items]  
    except Exception as e:
        traceback.print_exc()
        raise RuntimeError(f"Google Search error: {e}")

    if not items:
        return "I couldn't find any results for that search."

    # preparing prompt for LLM ────────────────────────────
    compact = [{"title": it["title"], "link": it["link"], "snippet": it["snippet"]} for it in items]

    prompt = (
        f"You are a travel expert.\n"
        f"Pick the top {how_many} activities someone should do based on the "
        f"Google search results below and present them in a {style} style. "
        "Give each activity a one‑sentence description.\n\n"
        f"{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


    # summerization with LLM ───────────────────────────────────
    summary = llm.invoke(prompt).content
    # Return as a dictionary with a tag (or using your message format) so that the router knows it is from tool.
    return {"content": summary, "from": "tool"}

from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

from google.genai import types
import google.genai as genai
client = genai.Client(api_key=GOOGLE_API_KEY)


# Define a helper to retry when per-minute quota is reached.
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]
    
import chromadb
import pandas as pd
import re
import openai
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ----------------------
# Initialize your embedding model.
# ----------------------
DB_NAME = "googlecardb"

embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="accommodation_reviews")

def llm_summarizer(text: str) -> str:
    """
    Summarize the input text using OpenAI's ChatCompletion endpoint.
    Adjust the model, max_tokens, and temperature parameters as needed.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Replace with your desired model if needed.
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Please summarize the following hotel reviews in a concise "
                        f"and friendly tone:\n\n{text}\n\nSummary:"
                    )
                }
            ],
            max_tokens=150,  # Adjust as needed for summary length.
            temperature=0.5,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error during summarization: {e}")
        return text

@tool("accommodation_search")
def accommodation_search(query: str, 
                         num_results: int = 3, 
                         style: str = "friendly", **kwargs) -> str:
    """
    Downloads a CSV of hotel reviews, loads (and indexes) the data into a ChromaDB vector database (if not already done),
    filters the dataset by the country found in the query, searches the database using the user's query, and returns
    accommodation recommendations.

    Args:
        query (str): The user's search query for accommodations (e.g., "find me hotels in Spain").
        num_results (int): The number of top search results to return.
        style (str): The output tone (e.g., "friendly").
    
    Returns:
        dict: A dictionary containing a summary of the search results.
    """
    try:
        # Step 1. Download and load the hotel reviews file.
        reviews_data = pd.read_csv(
            "Hotel_Reviews.csv",
            engine='python',         # Use a more forgiving CSV parser
            on_bad_lines='skip',     # Skip rows with too many fields
            encoding='utf-8',        # Ensure proper encoding
            na_values=['NA']         # Treat "NA" as a missing value
        )[['Hotel_Address', 'Positive_Review', 'Negative_Review', 'Hotel_Name']]

        # Step 1.5. Extract and apply a country filter from the query.
        # This regex looks for a phrase like "in Spain" and extracts "Spain".
        country = None
        country_match = re.search(r'\bin\s+([A-Za-z\s]+)', query)
        if country_match:
            country = country_match.group(1).strip()
            # Filter the reviews_data for rows where 'Hotel_Address' mentions the country.
            reviews_data = reviews_data[reviews_data['Hotel_Address'].str.contains(country, case=False, na=False)]
        
        if reviews_data.empty:
            return {"content": f"No hotels found in {country}." if country else "No hotels found.", "from": "tool"}
        
        # Step 2: Aggregate reviews for each unique hotel.
        # Aggregate reviews for each unique hotel, only using the first 10 reviews each for positive and negative reviews.
        grouped = reviews_data.groupby("Hotel_Name", as_index=False).agg({
            "Hotel_Address": "first",
            "Positive_Review": lambda x: " ".join(x.dropna().astype(str).head(10)),
            "Negative_Review": lambda x: " ".join(x.dropna().astype(str).head(10))
        })
        # Compute a review count for each hotel.
        review_counts = reviews_data.groupby("Hotel_Name").size().reset_index(name="review_count")
        
        # Merge in review counts.
        grouped = pd.merge(grouped, review_counts, on="Hotel_Name")
        
        # Select only the top 20 hotels (sorted by review count in descending order).
        top_hotels = grouped.sort_values("review_count", ascending=False).head(20)
        
        # Create a document per hotel using a list comprehension.
        results = [
            (
                str(i),
                "{} ({}):\nSummary of reviews: {}".format(
                    row["Hotel_Name"] if pd.notnull(row["Hotel_Name"]) else "Unknown Hotel",
                    row["Hotel_Address"] if pd.notnull(row["Hotel_Address"]) else "No address provided",
                    llm_summarizer(
                        "Positive Reviews: " + str(row["Positive_Review"]) +
                        "\nNegative Reviews: " + str(row["Negative_Review"])
                    )
                ),
                {
                    "hotel_name": row["Hotel_Name"] if pd.notnull(row["Hotel_Name"]) else "Unknown Hotel",
                    "hotel_address": row["Hotel_Address"] if pd.notnull(row["Hotel_Address"]) else "No address provided",
                    "review_count": row["review_count"]
                }
            )
            for i, row in top_hotels.iterrows()
        ]
        
        
        # Unpack the results into separate lists.
        ids, documents, metadatas = map(list, zip(*results))
            
        # Generate embeddings for only the filtered documents.
        embeddings = embed_fn.__call__(documents)
            
        # Add the documents to the ChromaDB collection.
        collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        
        # Step 3. Use the user's query to search the vector database.
        query_embedding = embed_fn.__call__([query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=num_results,
            # Assuming your collection returns metadata with the appropriate keys.
            include=["metadatas"]  
        )

        # Format a recommendation summary based on the returned metadata.
        # For instance, if results["metadatas"] is a list of lists for each query:
        recommendations_meta = results["metadatas"][0]  # For a single query.

        if recommendations_meta:
            summary_text = f"Based on your query '{query}', here are some accommodation recommendations:\n"
            for meta in recommendations_meta:
                hotel_name = meta.get("hotel_name", "Unknown Hotel")
                hotel_address = meta.get("hotel_address", "No address provided")
                review_count = meta.get("review_count", "N/A")
                summary_text += f"- {hotel_name} at {hotel_address} (Reviews: {review_count})\n"
        else:
            summary_text = "No matching accommodations found based on your preferences. If you would like you can try again with another preference or no preference."

        return {"content": summary_text, "from": "tool"}

    
    except Exception as e:
        traceback.print_exc()
        return {"content": f"Accommodation search error: {e}", "from": "tool"}

from langgraph.prebuilt import ToolNode

# Define the tools and create a "tools" node.
tools = [activity_search, accommodation_search]
tool_node = ToolNode(tools)

# Attach the tools to the model so that it knows what it can call.
llm_with_tools = llm.bind_tools(tools)


def maybe_route_to_tools(state: SearchState) -> str:
    """Route between chat and tool nodes if a tool call is made."""
    if not (msgs := state.get("messages", [])):
        raise ValueError(f"No messages found when parsing state: {state}")

    # Force human route if only the welcome message exists
    if len(msgs) == 1:
        return "human"
    
    msg = msgs[-1]

    if state.get("finished", False):
        # When a search is placed, exit the app. The system instruction indicates
        # that the chatbot should say thanks and goodbye at this point, so we can exit
        # cleanly.
        return END

    elif hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0:
        # Route to `tools` node for any automated tool calls first.
        if any(
            tool["name"] in tool_node.tools_by_name.keys() for tool in msg.tool_calls
        ):
            return "tools"
        else:
            return "searching"

    else:
        return "human"
    
def chatbot_with_tools(state: SearchState) -> SearchState:
    """The chatbot with tools. A simple wrapper around the model's own chat interface."""
    defaults = {"search": [], "finished": False}

    if state["messages"]:
        new_output = llm_with_tools.invoke([TRIPADVISOR_SYSINT] + state["messages"])
    else:
        new_output = AIMessage(content=WELCOME_MSG)

    # Set up some defaults if not already set, then pass through the provided state,
    # overriding only the "messages" field.
    return defaults | state | {"messages": [new_output]}

from langchain_core.messages.tool import ToolMessage

def search_node(state: SearchState) -> SearchState:
    """The searching node. This is where the search state is manipulated."""
    tool_msg = state.get("messages", [])[-1]
    search = state.get("search", [])
    outbound_msgs = []
    search_placed = False

    for tool_call in tool_msg.tool_calls:

        if tool_call["name"] == "activity_search":

            # Each search item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            if modifier_str:
                search.append(f'{tool_call["args"]} ({modifier_str})')
            else:
                search.append(tool_call["args"])
                              
            response = "\n".join(search)

        elif tool_call["name"] == "accommodation_search":
            # Each search item is just a string. This is where it assembled as "drink (modifiers, ...)".
            modifiers = tool_call["args"]["modifiers"]
            modifier_str = ", ".join(modifiers) if modifiers else "no modifiers"

            search.append(f'{tool_call["args"]} ({modifier_str})')
            response = "\n".join(search)
        else:
            raise NotImplementedError(f'Unknown tool call: {tool_call["name"]}')

        # Record the tool results as tool messages.
        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "search": search, "finished": search_placed}

graph_builder = StateGraph(SearchState)

# Add the nodes, including the new tool_node.
graph_builder.add_node("chatbot", chatbot_with_tools)
graph_builder.add_node("human", human_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("searching", search_node)

# Chatbot -> {searching, tools, human, END}
graph_builder.add_conditional_edges("chatbot", maybe_route_to_tools)
# Human -> {chatbot, END}
graph_builder.add_conditional_edges("human", maybe_exit_human_node)

# Tools always route back to chat afterwards.
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("searching", "chatbot")

graph_builder.add_edge(START, "chatbot")
graph_with_menu = graph_builder.compile()

# Image(graph_with_menu.get_graph().draw_mermaid_png())

state = graph_with_menu.invoke({"messages": []}, config)

