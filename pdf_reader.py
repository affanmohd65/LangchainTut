from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

## loading document
loader = TextLoader("docs.txt")
documents = loader.load()

## split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

## converting chunks into embedding & store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vectorstore = FAISS.from_documents(docs, embeddings)

## creating retriever
retriever = vectorstore.as_retriever()

## manually retrieving relevant documents
query="What are the key takeaways of this document?"
retrieved_doc=retriever.invoke(query)

## combine retrieved text into single prompt
retrieved_text = "\n".join([doc.page_content for doc in retrieved_doc])

## initialize llm
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash"
)

## manually pass retrieved text to llm
prompt=f"Based on following text, answer the question: {query}\n\n{retrieved_text}"
answer=llm.invoke(prompt)

## answer
print(answer)
