import traceback
import os
import requests
import json
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv.main import load_dotenv
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID= os.getenv("GOOGLE_CSE_ID")
GOOGLE_SEARCH_KEY= os.getenv("GOOGLE_SEARCH_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

google_search_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

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
    
    # print("search summary", compact)
    # print("items", items[0])


    # summerization with LLM ───────────────────────────────────
    summary = google_search_llm.invoke(prompt).content
    # Return as a dictionary with a tag (or using your message format) so that the router knows it is from tool.
    return {"content": summary, "from": "tool"}