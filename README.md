# TripoBot

An intelligent travel destination recommendation system that uses AI to analyze images and preferences to suggest personalized travel destinations.

## Installation

1. Clone this repository.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```bash
GOOGLE_API_KEY= ...
GOOGLE_SEARCH_KEY= ...
GOOGLE_CSE_ID= ...
```

## Usage

### Quick Start
Run the main flow to experience the complete recommendation system:
```bash
python flow.py
```

### Example Usage
You can also use the prewritten examples at the end of the `flow.py` to start.


## Project Structure

- `agents/` - Contains all agent implementations
- `flow.py` - Main execution flow

## Features

- Image-based destination matching
- Weather integration for accurate forecasts
- Budget-aware recommendations
- Customizable travel duration
- Accommodation preference consideration

## Links

- 📺 [YouTube Demo](https://www.youtube.com/watch?v=jO5xrYpYWhk)
- 📝 [Medium Article](https://medium.com/@sabadaftari/a-lang-graph-overview-to-design-a-tripobot-using-google-ai-capabilities-24c7f2120121)
- 🏆 [Kaggle Project](https://www.kaggle.com/code/sabadaftari/tripobot)


