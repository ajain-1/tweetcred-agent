# pylint: disable=import-error
import os
from fastapi import FastAPI, File, HTTPException, UploadFile, middleware
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from agent import get_structured_output, fact_check_tweet_image

from pydantic import BaseModel
import asyncio
import json


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/process_twitter/{username}")
async def process_twitter(username: str):
    username = username.strip()
    print(f"Processing Twitter username: {username}")
    if not username:
        raise HTTPException(status_code=400, detail="Invalid username provided.")

    # Run your program with the provided Twitter username
    gemini_output, pca_tweet = get_structured_output(username)

    # Clean the output to be actual JSON and remove newlines
    if isinstance(gemini_output, str):
        # Remove any newlines and extra whitespace from the string
        gemini_output = gemini_output.replace("\n", " ").strip()
        try:
            # Try to parse as JSON to ensure it's valid
            gemini_output = json.loads(gemini_output)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500, detail="Failed to parse output as JSON"
            ) from exc
    else:
        # Function to recursively clean string values in Python objects
        def clean_strings(obj):
            if isinstance(obj, dict):
                return {k: clean_strings(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_strings(item) for item in obj]
            elif isinstance(obj, str):
                return obj.replace("\n", " ").strip()
            else:
                return obj

        gemini_output = clean_strings(gemini_output)

    # Return the processed output along with the username in JSON format
    return JSONResponse(
        content={"username": username, "output": gemini_output, "pca_tweet": pca_tweet}
    )


@app.post("/fact_check/")  # add trailing slash
async def fact_check(file: UploadFile = File(...)):
    print(file)
    image_bytes = await file.read()

    print(image_bytes)

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image data provided.")
    print(f"Processing image of size: {len(image_bytes)} bytes")

    # delegate to the agent
    gemini_output, search_suggestions, source_links = fact_check_tweet_image(
        image_bytes
    )

    try:
        gemini_output = gemini_output.replace("\n", " ").strip()
        gemini_output = json.loads(gemini_output)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="Failed to parse output as JSON"
        ) from exc

    return JSONResponse(
        content={
            "output": gemini_output,
            "search_suggestions": search_suggestions,
            "source_links": source_links,
        }
    )


@app.get("/")
async def root():
    return JSONResponse(content={"message": "TweetCred"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8080)), workers=4
    )
