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