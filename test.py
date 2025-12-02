import subprocess
import sys

from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model


import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass("langchain API key")