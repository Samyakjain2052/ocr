# ocr.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import requests
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import time
import re
import os
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("id-analyzer-api")

# Set API keys
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "K88769549988957")
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Initialize FastAPI app
app = FastAPI(title="ID Card Age Analyzer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response model
class IDCardAnalysisResult(BaseModel):
    raw_text: str
    dob: Optional[str] = None
    age: Optional[int] = None
    is_minor: Optional[bool] = None

# Function to preprocess image
def preprocess_image(image_bytes):
    """Apply enhanced preprocessing to improve OCR accuracy"""
    try:
        # Convert bytes to numpy array
        image = Image.open(BytesIO(image_bytes))
        img = np.array(image.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        processed = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Clean up image with morphological operations
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to bytes
        processed_image = Image.fromarray(processed)
        buffer = BytesIO()
        processed_image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return image_bytes

# Extract text from image using OCR.space
def extract_text_from_image(image_bytes):
    """Extract text using OCR.space API"""
    try:
        # Preprocess image
        processed_bytes = preprocess_image(image_bytes)
        encoded_image = base64.b64encode(processed_bytes).decode("utf-8")
        
        # Advanced parameters for better accuracy
        payload = {
            "apikey": OCR_SPACE_API_KEY,
            "base64Image": f"data:image/png;base64,{encoded_image}",
            "language": "eng",
            "OCREngine": "2",
            "scale": "true",
            "detectOrientation": "true"
        }
        
        # Send request to OCR.space
        logger.info("Sending image to OCR.space for text extraction...")
        response = requests.post(OCR_SPACE_API_URL, data=payload, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"OCR.space API returned error: {response.status_code}")
            return None
            
        result = response.json()
        
        if "ParsedResults" in result and len(result["ParsedResults"]) > 0:
            return result["ParsedResults"][0]["ParsedText"]
        else:
            error_message = result.get("ErrorMessage", "Unknown error")
            logger.error(f"OCR error: {error_message}")
            return None
    
    except Exception as e:
        logger.error(f"Error in OCR process: {str(e)}")
        return None

# Extract DOB and calculate age using Groq API
def process_text_with_groq(text):
    """Extract DOB from text and calculate age using Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Enhanced prompt specifically focused on DOB extraction
        prompt = f"""
        You are analyzing text from an ID card. Extract ONLY the date of birth from this text and calculate the person's age:

        ```
        {text}
        ```

        Return a JSON object with:
        1. "dob": the exact date of birth as found (preserve original format)
        2. "age": the person's age in years based on today's date ({datetime.now().strftime('%Y-%m-%d')})
        3. "is_minor": boolean indicating if person is under 18

        If no date of birth is found, return null for dob and age, and false for is_minor.
        """
        
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a specialized system that extracts date of birth and calculates age from ID cards."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"}
        }
        
        logger.info("Sending text to Groq API for DOB extraction and age calculation...")
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        extracted_data = json.loads(result["choices"][0]["message"]["content"])
        
        return extracted_data
    
    except Exception as e:
        logger.error(f"Error processing with Groq: {str(e)}")
        return {"dob": None, "age": None, "is_minor": False}

# Single API endpoint that does everything
@app.post("/analyze", response_model=IDCardAnalysisResult)
async def analyze_id_card(file: UploadFile = File(...)):
    """
    Analyze ID card to extract text, find date of birth, and calculate age
    
    - **file**: Image file of an ID card
    
    Returns extracted text, date of birth, age, and minor status
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are allowed")
        
        # Read file
        file_bytes = await file.read()
        
        # Step 1: Extract text from image
        logger.info("Step 1: Extracting text from image...")
        raw_text = extract_text_from_image(file_bytes)
        
        if not raw_text:
            raise HTTPException(status_code=500, detail="Failed to extract text from image")
        
        logger.info(f"Text extracted: {raw_text[:100]}...")
        
        # Step 2: Process text with Groq to extract DOB and calculate age
        logger.info("Step 2: Processing text to extract DOB and calculate age...")
        result = process_text_with_groq(raw_text)
        
        # Step 3: Return the results
        response = {
            "raw_text": raw_text,
            "dob": result.get("dob"),
            "age": result.get("age"),
            "is_minor": result.get("is_minor", False)
        }
        
        logger.info(f"Analysis complete: DOB={result.get('dob')}, Age={result.get('age')}")
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error analyzing ID card: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing ID card: {str(e)}")

# Health check endpoint
@app.get("/")
async def read_root():
    return {"status": "online", "version": "1.0.0"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "127.0.0.1")  # Change to 0.0.0.0 for EC2
    
    logger.info(f"Starting ID Card Age Analyzer on {host}:{port}")
    uvicorn.run("ocr:app", host=host, port=port, reload=False)