import os
import traceback
import chromadb
import pandas as pd
import re
import openai
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.api_core import retry

from google.genai import types
import google.genai as genai

# from dotenv.main import load_dotenv
# load_dotenv()

# # --- Configuration ---
# GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
# if not GOOGLE_API_KEY:
#     raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


# client = genai.Client(api_key=GOOGLE_API_KEY)


# # Define a helper to retry when per-minute quota is reached.
# is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})


# class GeminiEmbeddingFunction(EmbeddingFunction):
#     # Specify whether to generate embeddings for documents, or queries
#     document_mode = True

#     @retry.Retry(predicate=is_retriable)
#     def __call__(self, input: Documents) -> Embeddings:
#         if self.document_mode:
#             embedding_task = "retrieval_document"
#         else:
#             embedding_task = "retrieval_query"

#         response = client.models.embed_content(
#             model="models/text-embedding-004",
#             contents=input,
#             config=types.EmbedContentConfig(
#                 task_type=embedding_task,
#             ),
#         )
#         return [e.values for e in response.embeddings]
# # ----------------------
# # Initialize your embedding model.
# # ----------------------
# DB_NAME = "googlecardb"

# embed_fn = GeminiEmbeddingFunction()
# embed_fn.document_mode = True

# chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection(name="accommodation_reviews")

# def llm_summarizer(text: str) -> str:
#     """
#     Summarize the input text using OpenAI's ChatCompletion endpoint.
#     Adjust the model, max_tokens, and temperature parameters as needed.
#     """
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",  # Replace with your desired model if needed.
#             messages=[
#                 {
#                     "role": "user",
#                     "content": (
#                         f"Please summarize the following hotel reviews in a concise "
#                         f"and friendly tone:\n\n{text}\n\nSummary:"
#                     )
#                 }
#             ],
#             max_tokens=150,  # Adjust as needed for summary length.
#             temperature=0.5,
#         )
#         return response.choices[0].message['content'].strip()
#     except Exception as e:
#         print(f"Error during summarization: {e}")
#         return text


# def accommodation_search(query: str, 
#                          num_results: int = 3, 
#                          style: str = "friendly", **kwargs) -> str:
#     """
#     Downloads a CSV of hotel reviews, loads (and indexes) the data into a ChromaDB vector database (if not already done),
#     filters the dataset by the country found in the query, searches the database using the user's query, and returns
#     accommodation recommendations.

#     Args:
#         query (str): The user's search query for accommodations (e.g., "find me hotels in Spain").
#         num_results (int): The number of top search results to return.
#         style (str): The output tone (e.g., "friendly").
    
#     Returns:
#         dict: A dictionary containing a summary of the search results.
#     """
#     try:
#         # Step 1. Download and load the hotel reviews file.
#         reviews_data = pd.read_csv(
#             "Hotel_Reviews.csv",
#             engine='python',         # Use a more forgiving CSV parser
#             on_bad_lines='skip',     # Skip rows with too many fields
#             encoding='utf-8',        # Ensure proper encoding
#             na_values=['NA']         # Treat "NA" as a missing value
#         )[['Hotel_Address', 'Positive_Review', 'Negative_Review', 'Hotel_Name']]

#         # Step 1.5. Extract and apply a country filter from the query.
#         # This regex looks for a phrase like "in Spain" and extracts "Spain".
#         country = None
#         country_match = re.search(r'\bin\s+([A-Za-z\s]+)', query)
#         if country_match:
#             country = country_match.group(1).strip()
#             # Filter the reviews_data for rows where 'Hotel_Address' mentions the country.
#             reviews_data = reviews_data[reviews_data['Hotel_Address'].str.contains(country, case=False, na=False)]
        
#         if reviews_data.empty:
#             return {"content": f"No hotels found in {country}." if country else "No hotels found.", "from": "tool"}
        
#         # Step 2: Aggregate reviews for each unique hotel.
#         # Aggregate reviews for each unique hotel, only using the first 10 reviews each for positive and negative reviews.
#         grouped = reviews_data.groupby("Hotel_Name", as_index=False).agg({
#             "Hotel_Address": "first",
#             "Positive_Review": lambda x: " ".join(x.dropna().astype(str).head(10)),
#             "Negative_Review": lambda x: " ".join(x.dropna().astype(str).head(10))
#         })
#         # Compute a review count for each hotel.
#         review_counts = reviews_data.groupby("Hotel_Name").size().reset_index(name="review_count")
        
#         # Merge in review counts.
#         grouped = pd.merge(grouped, review_counts, on="Hotel_Name")
        
#         # Select only the top 20 hotels (sorted by review count in descending order).
#         top_hotels = grouped.sort_values("review_count", ascending=False).head(20)
        
#         # Create a document per hotel using a list comprehension.
#         results = [
#             (
#                 str(i),
#                 "{} ({}):\nSummary of reviews: {}".format(
#                     row["Hotel_Name"] if pd.notnull(row["Hotel_Name"]) else "Unknown Hotel",
#                     row["Hotel_Address"] if pd.notnull(row["Hotel_Address"]) else "No address provided",
#                     llm_summarizer(
#                         "Positive Reviews: " + str(row["Positive_Review"]) +
#                         "\nNegative Reviews: " + str(row["Negative_Review"])
#                     )
#                 ),
#                 {
#                     "hotel_name": row["Hotel_Name"] if pd.notnull(row["Hotel_Name"]) else "Unknown Hotel",
#                     "hotel_address": row["Hotel_Address"] if pd.notnull(row["Hotel_Address"]) else "No address provided",
#                     "review_count": row["review_count"]
#                 }
#             )
#             for i, row in top_hotels.iterrows()
#         ]
        
        
#         # Unpack the results into separate lists.
#         ids, documents, metadatas = map(list, zip(*results))
            
#         # Generate embeddings for only the filtered documents.
#         embeddings = embed_fn.__call__(documents)
            
#         # Add the documents to the ChromaDB collection.
#         collection.add(
#                 ids=ids,
#                 documents=documents,
#                 embeddings=embeddings,
#                 metadatas=metadatas
#             )
        
#         # Step 3. Use the user's query to search the vector database.
#         query_embedding = embed_fn.__call__([query])
#         results = collection.query(
#             query_embeddings=query_embedding,
#             n_results=num_results,
#             # Assuming your collection returns metadata with the appropriate keys.
#             include=["metadatas"]  
#         )

#         # Format a recommendation summary based on the returned metadata.
#         # For instance, if results["metadatas"] is a list of lists for each query:
#         recommendations_meta = results["metadatas"][0]  # For a single query.

#         if recommendations_meta:
#             summary_text = f"Based on your query '{query}', here are some accommodation recommendations:\n"
#             for meta in recommendations_meta:
#                 hotel_name = meta.get("hotel_name", "Unknown Hotel")
#                 hotel_address = meta.get("hotel_address", "No address provided")
#                 review_count = meta.get("review_count", "N/A")
#                 summary_text += f"- {hotel_name} at {hotel_address} (Reviews: {review_count})\n"
#         else:
#             summary_text = "No matching accommodations found based on your preferences. If you would like you can try again with another preference or no preference."

#         return {"content": summary_text, "from": "tool"}

    
#     except Exception as e:
#         traceback.print_exc()
#         return {"content": f"Accommodation search error: {e}", "from": "tool"}


#         #method to index kone data ro load ()

#         #method call for query


from typing import Dict, Any, List, Optional
from state import TripState            # <- your dataclass
from dotenv.main import load_dotenv
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")


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

class AccommodationSearchAgent:
    """Callable node that returns {"response": <summary string>}"""

    # ── init & cached resources ─────────────────────────────────────────
    def __init__(self, csv_path: str = "Hotel_Reviews.csv"):
        self.csv_path = csv_path
        self.embed_fn = GeminiEmbeddingFunction()
        self.col      = chromadb.Client().get_or_create_collection("accommodation_reviews")
        self._df: Optional[pd.DataFrame] = None

    def _summarise_reviews(self, text: str) -> str:
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        "Please summarize the following hotel reviews in a "
                        "concise and friendly tone:\n\n" + text + "\n\nSummary:"
                    )
                }],
                max_tokens=150,
                temperature=0.5,
            )
            return rsp.choices[0].message["content"].strip()
        except Exception as e:
            print("Summarisation error:", e)
            return text[:200] + "…"

    # ───────────────────────────────────────────────────────────────────
    #  Private helpers
    # ───────────────────────────────────────────────────────────────────
    def _extract_query(self, state: TripState) -> str:
        """Pull the user’s search string from TripState."""
        return state.agent_input or (state.chat_history[-1] if state.chat_history else "")

    def _df_reviews(self) -> pd.DataFrame:
        """Lazy‑load and cache the CSV."""
        if self._df is None:
            self._df = pd.read_csv(
                self.csv_path, engine="python", on_bad_lines="skip",
                encoding="utf-8", na_values=["NA"]
            )[["Hotel_Address", "Positive_Review", "Negative_Review", "Hotel_Name"]]
        
        # Add a new column with the last 3 words
        self._df["CityCandidate"] = self._df["Hotel_Address"].str.split().apply(lambda x: " ".join(x[-3:]))

        return self._df.copy()
    
    def _detect_location_via_llm(self, user_query: str) -> str:
        """
        Send the user query to an LLM with a short prompt that 
        says "Extract the city/country from this text."
        Return the location string, or "" if none.
        """

        # If you prefer to use openai directly, do something like:
        try:
            # e.g. direct openai API call
            # The prompt is extremely minimal. Customize as needed:
            prompt = f"""You are a location extractor. 
            The user typed: '{user_query}'.
            Respond with exactly ONE city and ONE country name if you see it. 
            for example Barcelona Spain.
            If you see no location, respond with 'None'."""

            rsp = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # or your chosen model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            location_str = rsp.choices[0].message["content"].strip()
            if location_str.lower() == "none":
                return ""
            return location_str

        except Exception as e:
            print("Error calling location extraction LLM:", e)
            return ""
    
    def _filter_country(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Optional country filter via regex."""
        m = self._detect_location_via_llm(query)
        if m:
            country = m
            return df[df["CityCandidate"].str.contains(country, case=False, na=False)], country
        return df, None

    def _aggregate_hotels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse reviews into one row per hotel."""
        grouped = (
            df.groupby("Hotel_Name", as_index=False)
            .agg(
                Hotel_Address=("Hotel_Address", "first"),
                Positive_Review=("Positive_Review", lambda x: " ".join(x.dropna().head(5))),
                Negative_Review=("Negative_Review", lambda x: " ".join(x.dropna().head(5))),
            )
        )
        grouped["review_count"] = df.groupby("Hotel_Name").size().values
        return grouped.sort_values("review_count", ascending=False).head(20)

    def _index_top_hotels(self, top: pd.DataFrame):
        """Embed & add docs to Chroma."""
        docs, metas, ids = [], [], []
        for i, row in top.iterrows():
            summary = self._summarise_reviews(
                f"Positive: {row.Positive_Review}\nNegative: {row.Negative_Review}"
            )
            docs.append(f"{row.Hotel_Name} ({row.Hotel_Address}):\n{summary}")
            metas.append(
                {
                    "hotel_name": row.Hotel_Name,
                    "hotel_address": row.Hotel_Address,
                    "review_count": int(row.review_count),
                }
            )
            ids.append(str(i))

        emb = self.embed_fn(docs)
        self.col.add(ids=ids, documents=docs, embeddings=emb, metadatas=metas)

    def _similarity_search(self, query: str, n: int = 3) -> List[dict]:
        """Retrieve nearest hotels."""
        res = self.col.query(
            query_embeddings=self.embed_fn([query]),
            n_results=n,
            include=["metadatas"],
        )
        return res["metadatas"][0]
    def _format_response(self, query: str, hits: List[dict]) -> str:
        """Use LLM to summarize hotel recommendations into a nice paragraph."""
        if not hits:
            return "No matching accommodations found for your query."

        raw_summary = f"User query: {query}\n\nHotel options:\n"
        for h in hits:
            raw_summary += (
                f"- {h['hotel_name']} located at {h['hotel_address']} "
                f"with {h['review_count']} reviews.\n"
            )

        # Use LLM to turn this into a summary
        try:
            friendly_summary = self._summarise_reviews(raw_summary)
            return friendly_summary
        except Exception as e:
            print("LLM summarization failed:", e)
            return raw_summary  # Fallback to raw list if LLM fails

    # ───────────────────────────────────────────────────────────────────
    #  Public callable for LangGraph
    # ───────────────────────────────────────────────────────────────────
    def __call__(self, state: TripState) -> Dict[str, Any]:
        query = self._extract_query(state)
        if not query:
            return {"response": "I need a search query."}

        df, country = self._filter_country(self._df_reviews(), query)
        if df.empty:
            msg = f"No hotels found in {country}." if country else "No hotels found."
            return {"response": msg}

        top_hotels = self._aggregate_hotels(df)
        self._index_top_hotels(top_hotels)
        
        hits = self._similarity_search(query)
        return {"accommodation": self._format_response(query, hits)}
