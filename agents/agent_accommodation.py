import os
import chromadb
import pandas as pd
from google.api_core import retry
import google as genai
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from chromadb import Documents, EmbeddingFunction, Embeddings
from google.genai import types
from utils_agent import extract_json_from_response
from geopy.geocoders import Nominatim
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
    def __init__(self, csv_path: str = "Hotel_Reviews.csv"):
        self.csv_path = csv_path
        self.embed_fn = GeminiEmbeddingFunction()
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        self.col = chromadb.Client().get_or_create_collection("accommodation_reviews")
        self._df: Optional[pd.DataFrame] = None

    def _df_reviews(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.read_csv(
                self.csv_path, engine="python", on_bad_lines="skip",
                encoding="utf-8", na_values=["NA"]
            )[["Hotel_Address", "Positive_Review", "Negative_Review", "Hotel_Name"]]
            self._df["CityCandidate"] = self._df["Hotel_Address"].str.split().apply(lambda x: " ".join(x[-3:]))
        return self._df.copy()
    
    def _get_country_from_city(self, city: str) -> str:
        geolocator = Nominatim(user_agent="city-country-resolver")
        location = geolocator.geocode(city)
        if location:
            return location.address.split(",")[-1].strip()
        return "Country not found"

    def _filter_country(self, df: pd.DataFrame, query: Dict) -> pd.DataFrame:
        city=query['destination']
        country= self._get_country_from_city(city)
        parsed = city+" "+country
        return df[df["CityCandidate"].str.contains(parsed, case=False, na=False)]

    def _aggregate_hotels(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # use vectorized pandas ops
        docs, metas, ids = [], [], []
        top["doc"] = (
            top["Hotel_Name"] + " (" + top["Hotel_Address"] + "):\nPositive: " +
            top["Positive_Review"] + "\nNegative: " + top["Negative_Review"]
        )

        docs = top["doc"].tolist()
        metas = top[["Hotel_Name", "Hotel_Address", "review_count"]].rename(
            columns={"Hotel_Name": "hotel_name", "Hotel_Address": "hotel_address"}
        ).to_dict(orient="records")

        ids = top.index.astype(str).tolist()

        if not docs:
            return

        emb = self.embed_fn(docs)
        if not emb or len(emb) != len(docs):
            raise ValueError("Embedding failed or mismatch.")
        
        self.col.add(ids=ids, documents=docs, embeddings=emb, metadatas=metas)

    def _similarity_search(self, query: Dict, n: int = 3) -> List[dict]:
        destination = query.get("destination", "")
        preference = query.get("user_preference", "")
        full_query = f"Find hotels in {destination}. Preference: {preference}".strip()
        emb = self.embed_fn([full_query])
        res = self.col.query(query_embeddings=emb, n_results=n, include=["metadatas"])
        return res["metadatas"][0]


    def _format_response(self, query: Dict, hits: List[dict]) -> str:
        if not hits:
            return "No matching accommodations found for your query."
        text = f"User query: {query}\nHotel options:\n"
        for h in hits:
            text += f"- {h['hotel_name']} at {h['hotel_address']} (Reviews: {h['review_count']})\n"
        # Summarize with Gemini
        prompt = f"Please summarize the following hotel recommendations in a friendly tone:\n\n{text}\n\nSummary:"
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception as e:
            print("Summarization fallback:", e)
            return text

    def __call__(self, query: Dict) -> Dict[str, Any]:
        if not query:
            return {"response": "I need a search query."}
        query_json = extract_json_from_response(query)
        df = self._filter_country(self._df_reviews(), query_json)
        if df.empty:
            prompt = f"""
        You are a helpful travel assistant. A user is looking for hotels in **{query["destination"]}**.

        They specifically care about: **{query["user preference"]}**

        Return a short, friendly list of 3–5 hotels in **{query["destination"]}** that best match the user's preference. 
        For each hotel, include:
        - Hotel name
        - Neighborhood or address
        - Why it matches the preference

        Do not invent fake hotel names—use real or plausible examples.
        Answer in Markdown format.
        """
            return {"response": self.llm.invoke(prompt).content.strip()}
        else:
            top = self._aggregate_hotels(df)
            self._index_top_hotels(top)
            hits = self._similarity_search(query_json)
            return {"accommodation": self._format_response(query_json, hits)}

if __name__=="__main__":
    agent_inst = AccommodationSearchAgent()
    response = agent_inst("{'destination' : 'Paris', 'user preference' : 'good location for accommodation'}")
    print(response)
    # Explicitly close the genai client to prevent lingering threads
    # client.close()