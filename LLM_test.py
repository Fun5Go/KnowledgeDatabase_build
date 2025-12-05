from openai import OpenAI
from dotenv import load_dotenv
import requests
import os

load_dotenv()

# Load variables from .env into environment
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", "http://litellm.ame.local")

client = OpenAI(
    api_key=api_key,
    base_url=f"{base_url}/v1",  # adjust if your proxy uses a different path
)

response = client.chat.completions.create(
    model="azure/gpt-4.1",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in one short sentence."},
    ],
)

print(response.choices[0].message.content)