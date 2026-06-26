from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


## chat template

chat_template = ChatPromptTemplate([
    ('system','you are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
chat_history =[]


## load chat history

with open('history.txt') as f:
   chat_history.extend(f.readlines())

print(chat_history)

## create prompt

