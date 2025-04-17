import pydantic
from pydantic import BaseModel, Field


class Tweet(BaseModel):
    url: str
    text: str
    created_at: str
    favorite_count: int
    quote_count: int
    reply_count: int
    retweet_count: int
    is_quote_status: bool


class GeminiResponse(BaseModel):
    user_background: str
    credibility: int
    genuineness: int
    credibility_examples: list[Tweet]
    genuineness_examples: list[Tweet]
    credibility_explanation: str
    genuineness_explanation: str


class GeminiFactCheckResponse(BaseModel):
    rating: int
    rating_explanation: str
    rating_evidence: list[str]


class UserContext(BaseModel):
    background: str
