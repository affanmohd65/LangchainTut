from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="explain {topic}",
    input_variables=["topic"]
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chain = prompt | llm | StrOutputParser()

result=chain.invoke({
    "topic": "Blockchain"
})

print(result)