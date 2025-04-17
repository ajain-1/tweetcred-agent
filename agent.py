# pylint: disable=import-error

import os
from datetime import datetime, timezone
from exa_py import Exa
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sklearn.decomposition import PCA
import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .formats import (
    GeminiResponse,
    GeminiFactCheckResponse,
)

load_dotenv()

PROMPTS = {
    "EXA_PROMPT": r"""Get all posts on X (formerly Twitter) from this user: {USER}. """,
    "GEMINI_CONTEXT": r"""Research X user {USER_NAME}. If you can find who they are with 100% certainty, give a brief summary of who they are.""",
    "GEMINI_ANALYZE_TWEETS": r"""
Read this batch of {k} Tweets from {USER_NAME}. For each of the following categories, identify trends in the user's posts and give a rating from 0 to 10, where 10 means the user highly demonstrates that quality.
- Credibility: Are they always honest? Are their statements and predictions always verifiably true? Do they have credentials for the things they talk about?
- Genuineness: Do they post thoughtful opinions? Here are some ways that users are ungenuine:
    - Some users post in intentionally vague words, meant to sound deeply meaningful without actually saying anything
    - Some users always use exaggerating phrases in their posts, e.g. a news account that claims small developments are world-changing.
    - Some users are intentionally inflammatory and post controversial things just to increase engagement
    - Some users are just trying to sell or promote things for their own benefit, e.g. a self-proclaimed entrepreneur whose only business is selling an entrepreneurship course.

Here is context about the user: {USER_CONTEXT}
Here is context about the Tweets: {TWEETS}


Here are some sample ratings:

- Credibility 0: User believes Earth is flat.
- Credibility 5: User claimed fluoridated tap water is harmful, but has no background in biology or chemistry.
- Credibility 10: User has a math PhD and only posts textbook proofs of math theorems.

- Genuineness 0: Every post from this user is an identical link to a suspicious adult website.
- Genuineness 5: Almost every post from this user is supporting a particular political candidate. The user avoids lengthy debates and instead sends short insults to people who disagree.
- Genuineness 5: Every week, this user posts "This was a huge week in AI. Here's everything you need to know to catch up:..."
- Genuineness 10: Posts from this user are diverse and human. The user sometimes recounts embarrassing stories about themselves, and often admits when they don't know the answer to something.

Format your outputs like this:

<USER_BACKGROUND>: A paragraph with no line breaks that summarizes the user's background. 
<CREDIBILITY_EXPLANATION>: A paragraph with no line breaks that explains the user's credibility score.
<CREDIBILITY_EXAMPLES>: A list of tweets that demonstrate the user's credibility score. Each tweet should include the 
    url: str
    text: str
    created_at: str
    favorite_count: int
    quote_count: int
    reply_count: int
    retweet_count: int
    is_quote_status: bool
<CREDIBILITY>: A number from 0 to 10.
<GENUINENESS_EXPLANATION>: A paragraph with no line breaks that explains the user's genuineness score.
<GENUINENESS_EXAMPLES>: A list of tweets that demonstrate the user's credibility score. Each tweet should include the 
    url: str
    text: str
    created_at: str
    favorite_count: int
    quote_count: int
    reply_count: int
    retweet_count: int
    is_quote_status: bool
<GENUINENESS>: A number from 0 to 10.
""",
    "GEMINI_USER_CONTEXT": r"""Research X user {USER_NAME}. If you can find who they are with 100% certainty, give a detailed summary of who they are""",
    "GEMINI_STRUCTURE_REPONSE": r"""Below is a response from Gemini. Please format it into a JSON object as specified. Here is the text: 
    
    {text}""",
    "GEMINI_FACT_CHECK": r"""On a scale from 0 to 10, rate the truthfulness of this tweet, which is provided as an image so you have all the context. 0 is a complete lie, 10 is a completely accurate and well-proven truth, and 5 is a debated fact. 
    Search for reliable sources to support your evaluation. If it exists, decode the media in the tweet (such as an image) and include it in your reasoning argument. 
    Your explanation for the rating should be a paragraph with no line breaks and should focus on the specific tweet's content, not the tweet author or any tweets it is quoting.
    You must make every effort to use web search results in corroborating your rating and reasoning.
    
    Format your outputs like this:
    
    <RATING_EXPLANATION>: A paragraph with no line breaks that explains the rating and why it was given.
    <RATING_EVIDENCE>: A list of evidence pieces (each a single sentence or two) that support the rating. Each link should be a string. Separate this list with commas.
    <RATING>: A number from 0 to 10.""",
}

exa = Exa(api_key=os.getenv("EXA_API_KEY"))
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_tweets(user: str, start_time: str, end_time: str) -> list:
    prompt = PROMPTS["EXA_PROMPT"].format(USER=user)
    print(prompt)
    contents = exa.search_and_contents(
        query=prompt,
        text=True,
        start_published_date=start_time,
        end_published_date=end_time,
        include_domains=["x.com"],
        livecrawl="always",
        num_results=100,
        exclude_text=["t.co/"],
    ).results

    tweets = {}
    for result in contents:

        segments = [s.strip() for s in result.text.split("|")]
        status = segments[0]
        data = dict(
            segment.split(": ", 1) for segment in segments[1:] if ": " in segment
        )
        created_at = data.get("created_at")
        favorite_count = int(data.get("favorite_count", 0))
        quote_count = int(data.get("quote_count", 0))
        reply_count = int(data.get("reply_count", 0))
        retweet_count = int(data.get("retweet_count", 0))
        is_quote_status = data.get("is_quote_status", "False") == "True"
        lang = data.get("lang")

        tweet_data = {
            "url": result.url,
            # "title": result.title,
            # "score": result.score,
            # "author": result.author,
            "text": status,
            "created_at": created_at,
            "favorite_count": favorite_count,
            "quote_count": quote_count,
            "reply_count": reply_count,
            "retweet_count": retweet_count,
            "is_quote_status": is_quote_status,
            # "lang": lang,
        }

        if result.author == user and "/status/" in result.url:
            tweets[status] = tweet_data

    return tweets.values()


# Return the tweet closest to the first principal component of a list of tweets
def get_principal_tweet(tweets: list[dict]) -> dict:
    """
    Find the tweet that is closest to the first principal component of all tweet embeddings.

    Args:
        tweets: A list of tweet dictionaries containing text and metadata

    Returns:
        The tweet dictionary closest to the first principal component
    """
    # Handle empty tweets list
    if not tweets:
        return {"text": "No tweets available to analyze."}

    # Extract the text from the tweets and embed them
    texts = [tweet["text"] for tweet in tweets if "t.co" not in tweet["text"]]
    embeddings = np.vstack([embed_model.get_text_embedding(t) for t in texts])

    # Run PCA, get first principal component
    pca = PCA(n_components=1)
    pca.fit(embeddings)
    pc1 = pca.components_[0]  # direction
    mean_v = pca.mean_  # centroid

    # Compute orthogonal distance to PC line
    diff = embeddings - mean_v
    proj_lens = diff.dot(pc1)
    proj_pts = mean_v + np.outer(proj_lens, pc1)
    dists = np.linalg.norm(embeddings - proj_pts, axis=1)

    # Find the dict whose text-embedding is closest to PC
    idx = int(np.argmin(dists))
    closest_entry = tweets[idx]

    return closest_entry


def get_all_tweets(user: str) -> list:

    all_tweets = []

    for segment in range(3):
        year = 2025 - segment  # Determine the year for the current segment

        start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        start_date_iso = start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        end = datetime(year, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc)
        end_date_iso = end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        print(start_date_iso, "IS NOT EQUAL TO", end_date_iso)
        tweets = get_tweets(user, start_date_iso, end_date_iso)
        all_tweets.extend(tweets)

    return all_tweets


def textualize_tweets(tweets: list) -> str:
    text = ""
    for tweet in tweets:
        for key, value in tweet.items():
            text += f"{key}: {value}\n"
        text += "\n"

    return text


def gemini_generate(prompt, schema=None):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    if schema is None:
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.5,
        )
    else:
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema
        )

    results = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        results += chunk.text

    return results


def get_user_context(user: str) -> str:
    return gemini_generate(PROMPTS["GEMINI_USER_CONTEXT"].format(USER_NAME=user))


# ENTRY POINT
def get_structured_output(USER):

    USER = USER.strip()

    user_context = get_user_context(USER)

    all_tweets = get_all_tweets(USER)
    all_tweets_text = textualize_tweets(all_tweets)

    prompt = PROMPTS["GEMINI_ANALYZE_TWEETS"].format(
        k=len(all_tweets),
        USER_NAME=USER,
        USER_CONTEXT=user_context,
        TWEETS=all_tweets_text,
    )

    output = gemini_generate(prompt)
    gemini_output = gemini_generate(
        PROMPTS["GEMINI_STRUCTURE_REPONSE"].format(text=output), schema=GeminiResponse
    )

    print("GEMINI OUTPUT\n")
    print(gemini_output)

    llama_index_output = get_principal_tweet(all_tweets)

    return gemini_output, llama_index_output


def fact_check_tweet_image(image_bytes: bytes):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PROMPTS["GEMINI_FACT_CHECK"]),
                types.Part.from_bytes(mime_type="image/png", data=image_bytes),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )

    # Get the full response first (not streaming)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    results = response.candidates[0].content.parts[0].text

    # Initialize lists for source links and search suggestions
    source_links = []
    search_suggestions = []

    # Extract source links if available

    print("SOURCE LINKS\n")
    print(response.candidates[0].grounding_metadata)

    if response.candidates[0].grounding_metadata.grounding_chunks is not None:
        for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
            if (
                hasattr(chunk, "web")
                and chunk.web is not None
                and hasattr(chunk.web, "uri")
                and chunk.web.uri is not None
            ):
                print("LINKS")
                print(chunk.web.uri)
                title = None
                if hasattr(chunk.web, "title") and chunk.web.title is not None:
                    title = chunk.web.title
                source_links.append(
                    {
                        "uri": chunk.web.uri,
                        "title": title,
                    }
                )

    # Get Google Search suggestions if available
    if (
        hasattr(response.candidates[0], "grounding_metadata")
        and response.candidates[0].grounding_metadata is not None
    ):
        if (
            hasattr(response.candidates[0].grounding_metadata, "web_search_queries")
            and response.candidates[0].grounding_metadata.web_search_queries is not None
        ):
            search_suggestions = response.candidates[
                0
            ].grounding_metadata.web_search_queries

    gemini_output = gemini_generate(
        PROMPTS["GEMINI_STRUCTURE_REPONSE"].format(text=results),
        schema=GeminiFactCheckResponse,
    )

    print("GEMINI OUTPUT\n")
    print(gemini_output)

    return gemini_output, search_suggestions, source_links


def main():
    # user_context = get_user_context(USER)
    # print("Got user context\n")
    # all_tweets = get_all_tweets(USER)
    # print("Got all tweets\n")
    # all_tweets_text = textualize_tweets(all_tweets)

    # principal_tweet = get_principal_tweet(all_tweets)
    # print("Got PCA")

    # print("PRINCIPAL TWEET")
    # print(principal_tweet["text"])
    # print("ALL TWEETS")
    # for i in range(min(10, len(all_tweets))):
    #     print(all_tweets[i])
    #     print("")
    pass


if __name__ == "__main__":
    main()
