from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

import tools

agent =ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)