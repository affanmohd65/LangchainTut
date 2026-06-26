from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = GoogleGenerativeAI(model="gemini-2.5-flash")

messages = [

    SystemMessage(content='You are helpful assistant'),
    HumanMessage(content='tell me about langchain')
]
result = model.invoke(messages)
messages.append(AIMessage(content=result))

print(messages)