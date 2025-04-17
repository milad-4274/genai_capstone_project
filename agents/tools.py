from datetime import datetime, timedelta
from typing import Dict, Union
import requests
from urllib.parse import quote
# from langchain_core.tools import tool

# @tool
def get_weather(city: str, date_str: str) -> Union[Dict[str, Union[str, float]], str]:
    """
    Get the weather forecast for a specific city and date.

    Args:
        city (str): City name (e.g., "Paris")
        date_str (str): Date in YYYY-MM-DD format

    Returns:
        A dictionary with min and max temperature and weather summary,
        or a string describing an error.
    """
    try:
        if isinstance(date_str, datetime):
            target_date = date_str.date()
        elif isinstance(date_str, str):
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            return "Error: Provided date must be a string or datetime object."

        if target_date < datetime.now().date():
            return "Error: Date cannot be in the past."

    except Exception as e:
        return f"Error parsing date: {e}"


    # Get coordinates using OpenStreetMap
    try:
        encoded_city = quote(city[0])
        geo_url = f"https://nominatim.openstreetmap.org/search?q={encoded_city}&format=jsonv2&limit=1"
        headers = {'User-Agent': 'WeatherForecastApp/1.0'}
        geo_res = requests.get(geo_url, headers=headers)
        geo_res.raise_for_status()
        geo_data = geo_res.json()
        if not geo_data:
            return f"Error: Could not find coordinates for '{city}'"
        lat = float(geo_data[0]['lat'])
        lon = float(geo_data[0]['lon'])
    except Exception as e:
        return f"Error getting coordinates: {e}"

    # Get weather data from 7timer API
    try:
        url = f"http://www.7timer.info/bin/api.pl?lon={lon}&lat={lat}&product=civil&output=json"
        weather_res = requests.get(url)
        weather_res.raise_for_status()
        weather_data = weather_res.json()
    except Exception as e:
        return f"Error fetching weather data: {e}"

    # Parse forecast data
    results = []
    for item in weather_data.get("dataseries", []):
        timepoint = item["timepoint"]
        base_date = datetime.strptime(str(weather_data["init"]), "%Y%m%d%H")
        forecast_time = base_date + timedelta(hours=timepoint)
        if forecast_time.date() == target_date:
            results.append({
                "temp": item["temp2m"],
                "weather_desc": item["weather"],
                "rh": int(item["rh2m"].strip('%')) if "rh2m" in item else None,
                "wind": item.get("wind10m", {}).get("speed", None)
            })

    if not results:
        return f"No forecast data available for {target_date} in {city}"

    min_temp = min(r["temp"] for r in results)
    max_temp = max(r["temp"] for r in results)
    weather_summary = _get_most_frequent_weather(results)

    return {
        "city": city,
        "date": str(target_date),
        "min_temp": min_temp,
        "max_temp": max_temp,
        "weather_summary": weather_summary
    }

def _get_most_frequent_weather(results):
    freq = {}
    for r in results:
        w = r.get("weather_desc", "Unknown")
        freq[w] = freq.get(w, 0) + 1

    return max(freq, key=freq.get) if freq else "Unknown"

