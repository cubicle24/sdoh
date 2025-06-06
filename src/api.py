#!/usr/bin/env python3
"""
SDOH Risk Factors API Service

This module provides web service for extracting SDOH risk factors from a clinical note
"""

from ast import DictComp
import logging
import json
import os
import time
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from sdoh_agent import run_agent, AgentState


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sdoh_api")

# Initialize the SDOH agent
try:
    # note = open("../clinical_notes_repository/note_1_medicine_CHF.txt").read()
    # note = open("../clinical_notes_repository/easy_note.txt").read()
    note = open("../clinical_notes_repository/psychiatric_note.txt").read()
    start_state: AgentState = {
        "note": note,
        "sdoh": {},
        "intervention": {},
        "retry_count" : 0,
        "zipcode_tool_called" : False,
        "zipcode" : "",
        "social_services" : [],
        "audit_trail" : []
    }

    # end_state = run_agent(note, start_state)
    # pprint(f"Final state: {end_state}")
    # logger.info("Started sdoh agent")
except Exception as e:
    logger.error(f"Failed to read in note, couldn't start SDOH agent: {str(e)}")
    raise

# Create FastAPI app
app = FastAPI(
    title="Clinical Guidelines API",
    description="API for generating screening recommendations based on clinical notes",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoteRequest(BaseModel):
    note: str = Field(
        ...,
        description="Patient clinical note text",
        min_length=10
    )
    class Config:
        json_schema_extra = {
            "example": {
                "note": "Patient is a 68 year old male. He is homeless and lives in his car. He lost his job recently and relies on meals from churches. His car also broke down."
            }
        }

class SDOHResponse(BaseModel):
    sdoh: Dict[str, Dict]
    audit_trail: Dict[str, List]
    
    class Config:
        schema_extra = {
            "example": {
            "sdoh": {
                "housing_instability": {"present": false, "reasoning": "No mention of housing issues.", "z_code": [], "interventions": []},
                "food_insecurity": {"present": false, "reasoning": "No mention of food insecurity.", "z_code": [], "interventions": []},
                "lack_of_transportation": {"present": false, "reasoning": "No mention of transportation issues.", "z_code": [], "interventions": []},
                "financial_hardship": {"present": false, "reasoning": "No mention of financial issues.", "z_code": [], "interventions": []},
                "domestic_violence": {"present": true, "reasoning": "Patient reports hitting her husband.", "z_code": ["Z62.81"], "interventions": ["Contact the Domestic Violence Community Advocacy Program at Salvation Army Eastside Corps for support and resources.", "Consider counseling or therapy services specializing in domestic violence.", "Explore local shelters or safe houses if immediate safety is a concern."]},
                "language_barriers": {"present": false, "reasoning": "No mention of language barriers.", "z_code": [], "interventions": []},
                "low_health_literacy": {"present": true, "reasoning": "Patient unable to specify therapy learnings.", "z_code": ["Z55.9"], "interventions": ["Provide educational materials in simpler language.", "Offer one-on-one health education sessions to improve understanding of health conditions and treatments.", "Utilize teach-back methods to ensure comprehension of health information."]}
            },
            "audit_trail": [
                {
                "step": "extract_sdoh_risk_factors",
                "timestamp": "2025-06-05T19:11:23.099026",
                "changes": {
                    "modified": {
                    "sdoh": {
                        "housing_instability": {"present": false, "reasoning": "No mention of housing issues."},
                        "food_insecurity": {"present": false, "reasoning": "No mention of food insecurity."},
                        "lack_of_transportation": {"present": false, "reasoning": "No mention of transportation issues."},
                        "financial_hardship": {"present": false, "reasoning": "No mention of financial issues."},
                        "domestic_violence": {"present": true, "reasoning": "Patient reports hitting her husband."},
                        "language_barriers": {"present": false, "reasoning": "No mention of language barriers."},
                        "low_health_literacy": {"present": true, "reasoning": "Patient unable to specify therapy learnings."}
                    }
                    },
                    "added": {},
                    "removed": {}
                }
                },
                {
                "step": "map_to_z_codes",
                "timestamp": "2025-06-05T19:11:27.578729",
                "changes": {
                    "modified": {
                    "sdoh": {
                        "housing_instability": {"present": false, "reasoning": "No mention of housing issues.", "z_code": []},
                        "food_insecurity": {"present": false, "reasoning": "No mention of food insecurity.", "z_code": []},
                        "lack_of_transportation": {"present": false, "reasoning": "No mention of transportation issues.", "z_code": []},
                        "financial_hardship": {"present": false, "reasoning": "No mention of financial issues.", "z_code": []},
                        "domestic_violence": {"present": true, "reasoning": "Patient reports hitting her husband.", "z_code": ["Z62.81"]},
                        "language_barriers": {"present": false, "reasoning": "No mention of language barriers.", "z_code": []},
                        "low_health_literacy": {"present": true, "reasoning": "Patient unable to specify therapy learnings.", "z_code": ["Z55.9"]}
                    }
                    },
                    "added": {},
                    "removed": {}
                }
                },
                {
                "step": "get_zipcode",
                "timestamp": "2025-06-05T19:11:28.406223",
                "changes": {
                    "modified": {},
                    "added": {"zipcode": "98029", "zipcode_tool_called": true},
                    "removed": {}
                }
                },
                {
                "step": "search_social_services",
                "timestamp": "2025-06-05T19:11:37.789632",
                "changes": {
                    "modified": {},
                    "added": {
                    "social_services": [
                        {
                        "name": "Issaquah Community Services",
                        "address": "180 East Sunset Way, Issaquah, WA 98027",
                        "phone": "425-837-3125",
                        "website": "https://www.issaquahcommunityservices.org/",
                        "services": ["Emergency rent assistance", "Utility assistance"]
                        },
                        {
                        "name": "Issaquah Food & Clothing Bank",
                        "address": "179 1st Ave SE, Issaquah, WA 98027",
                        "phone": "425-392-4123",
                        "website": "https://issaquahfoodbank.org/",
                        "services": ["Food assistance", "Clothing bank", "Hygiene and household necessities", "Financial assistance in partnership with Issaquah Community Services"]
                        },
                        {
                        "name": "Hopelink",
                        "address": "11011 120th Ave NE, Kirkland, WA 98033",
                        "phone": "425-869-6000",
                        "website": "https://www.hopelink.org/",
                        "services": ["Food banks", "Energy assistance", "Affordable housing", "Family development programs", "Transportation services", "Adult education programs"]
                        },
                        {
                        "name": "Salvation Army Eastside Corps",
                        "address": "911 164th Ave NE, Bellevue, WA 98008",
                        "phone": "425-452-7300",
                        "website": "https://bellevue.salvationarmy.org/",
                        "services": ["Emergency food assistance", "Utility assistance", "Domestic violence community advocacy program"]
                        }
                    ],
                    "social_services_tool_called": true
                    },
                    "removed": {}
                }
                },
                {
                "step": "recommend_interventions",
                "timestamp": "2025-06-05T19:11:44.001710",
                "changes": {
                    "modified": {
                    "sdoh": {
                        "housing_instability": {"present": false, "reasoning": "No mention of housing issues.", "z_code": [], "interventions": []},
                        "food_insecurity": {"present": false, "reasoning": "No mention of food insecurity.", "z_code": [], "interventions": []},
                        "lack_of_transportation": {"present": false, "reasoning": "No mention of transportation issues.", "z_code": [], "interventions": []},
                        "financial_hardship": {"present": false, "reasoning": "No mention of financial issues.", "z_code": [], "interventions": []},
                        "domestic_violence": {"present": true, "reasoning": "Patient reports hitting her husband.", "z_code": ["Z62.81"], "interventions": ["Contact the Domestic Violence Community Advocacy Program at Salvation Army Eastside Corps for support and resources.", "Consider counseling or therapy services specializing in domestic violence.", "Explore local shelters or safe houses if immediate safety is a concern."]},
                        "language_barriers": {"present": false, "reasoning": "No mention of language barriers.", "z_code": [], "interventions": []},
                        "low_health_literacy": {"present": true, "reasoning": "Patient unable to specify therapy learnings.", "z_code": ["Z55.9"], "interventions": ["Provide educational materials in simpler language.", "Offer one-on-one health education sessions to improve understanding of health conditions and treatments.", "Utilize teach-back methods to ensure comprehension of health information."]}
                    }
                    },
                    "added": {},
                    "removed": {}
                }
                },
                {
                "step": "end",
                "timestamp": "2025-06-05T19:11:44.002158",
                "changes": {"modified": {}, "added": {}, "removed": {}}
                }
            ]
            }
        }

        

# Middleware for request timing and logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request {request.method} {request.url.path} processed in {process_time:.4f} seconds")
    return response

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# API endpoints

@app.post("/sdoh/run", response_model=RecommendationResponse)
async def get_recommendations(request: ClinicalNoteRequest):
    """
    Generate screening recommendations based on a clinical note.
    Returns patient data extracted from the note and recommended screening tests.
    """
    try:
        logger.info("Processing recommendation request")
        raw_response = guidelines_system.generate_recommendations({"clinical_note": request.clinical_note})
        
        # Parse the response if it's a string
        if isinstance(raw_response["recommendations"], str):
            try:
                recommendations = json.loads(raw_response["recommendations"])
                raw_response["recommendations"] = recommendations
            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")
                logger.error(f"LLM response: {raw_response['recommendations']}")
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="LLM returned invalid JSON format"
                )
                
        logger.info("Successfully generated recommendations")
        return raw_response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/")
async def health_check():
    """Tests to verify the API is running."""
    return {"status": "hello SDOH api is up", "time": time.time()}

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Health check called: Guidelines API server on {host}:{port}")
    uvicorn.run(
        "api:app", 
        host=host, 
        port=port, 
        reload=False,  # Set to False in production
        log_level="info"
    )