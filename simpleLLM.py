from langchain import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

## prompt templae
prompt = PromptTemplate(
    input_variables = ['topic'],
    template = "suggest 5 interesting facts about {topic}"
)

## input
topic = input('Enter a topic')

##using prompt template
formatted_input = prompt.format(topic=topic)

##calling llm
title = llm.predict(formatted_input)

###print the output
print("generated title: ", title)
