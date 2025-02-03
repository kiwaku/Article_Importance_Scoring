import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

API_URL = "https://api.groq.com/openai/v1/chat/completions"
API_KEY = os.getenv("API_KEY")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def summarize_article_from_file(json_filepath):
    """
You are a concise AI. Read the user’s article and:

1. Write a short paragraph summarizing the main facts, the language must be absolutely objective, minimal, and factual.
2. Then list all events in a structured format (Actor, Action, Object, Time, Location, Quantity, Source, Confidence, Additional Context).

If a field is missing, write "UNKNOWN".
    """
    # 1. Load the JSON file
    with open(json_filepath, 'r', encoding='utf-8') as file:
        article = json.load(file)
    
    # 2. Extract relevant fields from the JSON
    title = article.get("title", "UNKNOWN")
    source_domain = article.get("source_domain", "UNKNOWN")
    date_publish = article.get("date_publish", "UNKNOWN")
    maintext = article.get("maintext", "UNKNOWN")
    
    # 3. Build the user_content by including any metadata you want the model to see
    #    but keep it concise to save tokens
    user_content = (
        f"Article Title: {title}\n"
        f"Source: {source_domain}\n"
        f"Published: {date_publish}\n\n"
        f"Main Text:\n{maintext}"
    )
    
    # 4. Construct the system message that instructs the model to produce both
    #    the short summary and the structured JSON events
    system_message = (
        "You are a concise AI system that reads the given article. "
        "First, produce a short, natural-language summary (like a mini-article) that includes: "
        "  - A brief title or heading "
        "  - 1~2 paragraphs summarizing the main points. "
        "Keep it objective, minimal, and factual. "
        "\n\n"
        "Then, output a structured JSON array called 'events', where each element has these fields: "
        "actor, action, object, time, location, quantity, source, confidence, additional_context. "
        "If any field is not found in the text, set it to 'UNKNOWN'. "
        "If the text describes multiple events, output multiple JSON objects in the array. "
        "Ensure the JSON is valid and that the summary is separate, not inside the JSON. "
        "No speculation or guessing—only use information found in the article. "
    )
    
    # 5. Prepare the payload for the API request
    data = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ],
        "temperature": 0.2,
        "max_tokens": 500  # Increase if you need more space for both summary + JSON
    }
    
    # 6. Send the request to the API
    response = requests.post(API_URL, json=data, headers=headers)
    response.raise_for_status()
    
    # 7. Print the output (summary + JSON) to the console
    output = response.json()["choices"][0]["message"]["content"]
    print(output)

# Usage example:
if __name__ == "__main__":
    # Provide the path to your JSON file containing the article
    json_file_path = "news_world-australia-51289897_1738340100.html.json"
    summarize_article_from_file(json_file_path)