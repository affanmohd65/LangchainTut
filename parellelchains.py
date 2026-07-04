from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)
model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt1 = PromptTemplate(
    template='Generate short and simple notes about \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="""
    Generate 5 quiz questions from the following text.{text}""",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="""Create a document with two sections.SECTION 1: NOTES{notes}
                SECTION 2: QUIZ {quiz} Do not rewrite or remove the quiz. Preserve all quiz questions exactly.""",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chains = RunnableParallel(
    notes=prompt1 | model1 | parser,
    quiz=prompt2 | model2 | parser
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chains | merge_chain 

text = """
Blockchain is an immutable digital ledger that enables secure transactions across a peer-to-peer network. It records, stores and verifies data using decentralized techniques to eliminate the need for third parties, like banks or governments. Every transaction is recorded and stored in a block on the blockchain. Each block is encrypted for protection and chained to the preceding block, establishing a code-based chronological order. This means that data stored on a blockchain cannot be deleted or modified without consensus of a network. These new-age databases act as a single source of truth and facilitate trustless and transparent data exchange among an interconnected network of computers.

Apart from moving cryptocurrencies from one wallet to the next, blockchain technology is an emerging technology with wide-ranging application potential, from preventing fraudulent banking and supply-chain bottlenecks to safeguarding medical records.
Blockchain is a revolutionary technology because it helps reduce security risks, stamp out fraud and bring transparency in a scalable way.

Popularized by its association with cryptocurrency and non-fungible tokens (NFTs), blockchain technology has since evolved to become a management solution for all types of global industries. Blockchain technology can be found providing transparency for the food supply chain, securing healthcare data, innovating gaming and changing how we handle data and ownership on a large scale.

One of the most important concepts in blockchain technology is decentralization. No one computer or organization can own the chain. Instead, it is a distributed ledger via the nodes connected to the chain. Blockchain nodes can be any kind of electronic device that maintains copies of the chain and keeps the network functioning.

Every node has its own copy of the blockchain and the network must algorithmically approve any newly mined block for the chain to be updated, trusted and verified. Since blockchains are transparent, every action in the ledger can be easily checked and viewed, creating inherent blockchain security. Each participant is given a unique alphanumeric identification number that shows their transactions.

Combining public information with a system of checks and balances helps the blockchain maintain integrity and creates trust among users. Essentially, blockchains can be thought of as the scalability of trust via technology.


"""
result = chain.invoke({'text': text})
print(result)
chain.get_graph().print_ascii()