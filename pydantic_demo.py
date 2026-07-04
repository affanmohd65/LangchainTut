from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Annotated

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

#schema

class Review(BaseModel):

    key_themes: list[str] = Field(description = "write down all key themes discussed in review.")
    summary: str = Field(description = "A brief summary of review")
    sentiment:Literal["positive", "negative", "neutral"] = Field(description = "Return the sentiment either negative, positive or neutral")
    pros:Optional[List[str]] = Field (default=None,description = "List of pros mentioned in review.")
    cons:Optional[List[str]] = Field (default=None, description = "List of cons mentioned in review.")

structured_model = model.with_structured_output(Review)


result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Jack
""")

print(result)

## Accessing individual fields
print(result.summary)

## convert to dictionary
print(result.model_dump())

### convert to json
print(result.model_dump_json(indent=2))
