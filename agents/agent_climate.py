import os
import requests
import google.generativeai as genai
from PIL import Image
import io
from dotenv.main import load_dotenv

load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Setup Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def download_image_bytes(image_url: str) -> Image.Image:
    """Download image from URL and return raw bytes."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(image_url, headers=headers)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


def climate_destination_recommender(image_url: str, 
                                    style: str = "friendly", 
                                    month: str = "April", **kwargs) -> str:
    """
    Uses Gemini 2.0-flash=exp to interpret the climate/activity from an image (local or URL),
    then recommends similar destinations for that type of environment in a specific month.
    """
    try:
        image_bytes = download_image_bytes(image_url)

        prompt = (
            f"This is a photo of a place or climate. Based on this image, describe:\n"
            f"1. What kind of location or activity this seems to be (e.g., ski resort, beach, hiking trail).\n"
            f"2. What kind of weather is represented.\n"
            f"3. Suggest one or two destinations with similar climate in {month}.\n"
            f"4. Include expected temperatures and recommended activities.\n"
            f"5. Write in a {style} tone."
        )

        EXAMPLES = [
            {
                "img": download_image_bytes("https://upload.wikimedia.org/wikipedia/commons/9/9d/Seychelles_Beach.jpg"),
                "month": "July",
                "style": "friendly",
                "answer": (
                    "1. A tranquil tropical beach perfect for sun‑lounging and gentle swimming.\n"
                    "2. Clear skies, bright sun, 28 °C with a light sea breeze.\n"
                    "3. Try Bora Bora or the Maldives in July.\n"
                    "4. Expect 27‑30 °C; plan for snorkeling, paddle‑boarding, and sunset cruises.\n"
                    "5. Friendly tone 🙂: *Pack reef‑safe sunscreen and a good book; paradise awaits!*"
                ),
            },
            {
                "img": download_image_bytes("https://upload.wikimedia.org/wikipedia/commons/0/03/Panorama_vom_Gornergrat-Zermatt.jpg"),
                "month": "January",
                "style": "adventurous",
                "answer": (
                    "1. A high‑alpine ski resort with well‑groomed downhill runs.\n"
                    "2. Crisp winter weather: −5 °C, light powder snow, low humidity.\n"
                    "3. Consider Whistler (Canada) or St. Anton (Austria) in January.\n"
                    "4. Temps −10 °C to −2 °C; carve fresh pistes, try night skiing, warm up with après‑ski fondue.\n"
                    "5. Adventurous tone 🏂: *Strap in, breathe the icy air, and chase that first‑tracks adrenaline!*"
                ),
            },
            ]
        # Build the content list:  (ex1_img, ex1_answer, ex2_img, ex2_answer, target_img, prompt)
        content = []
        for ex in EXAMPLES:
            content.extend([ex["img"], ex["answer"]])
        content.extend([image_bytes, prompt])

        response = model.generate_content(content)

        return response.text.strip()

    except Exception as e:
        return f"❌ Error: {e}"