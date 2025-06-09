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
from fastapi.responses import JSONResponse, StreamingResponse
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
    graph = build_agent(start_state)
except Exception as e:
    logger.error(f"Failed to initialize SDOH agent: {str(e)}")
    raise HTTPException(status_code=500, detail="Sorry, our fault. Could not start SDOH agent")


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

class ClinicalNoteRequest(BaseModel):
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
                "domestic_violence": {"present": true, "reasoning": "Patient reports being hit by her husband.", "z_code": ["Z62.81"], "interventions": ["Contact the Domestic Violence Community Advocacy Program at Salvation Army Eastside Corps for support and resources.", "Consider counseling or therapy services specializing in domestic violence.", "Explore local shelters or safe houses if immediate safety is a concern."]},
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
        content={"detail": "We're sorry, an unexpected internal server error happened. Please try again later."}
    )

#user friendly messages when showing reasoning steps
NODES_DICT = {
    "extract_sdoh_risk_factors": "Finding SDOH risk factors",
    "map_to_z_codes": "Matching SDOH risk factors to ICD-10 Z codes",
    "get_zipcode": "Looking for patient's zipcode",
    "search_social_services": "Searching for nearby social services",
    "recommend_interventions": "Designing custom interventions for the patient",
}

# API endpoints
@app.post("/sdoh/run_agent", response_model=SDOHResponse)
async def parse_sdoh_and_make_recs(note: ClinicalNoteRequest) -> StreamingResponse:
    """
    Parses a note and streams its step by step reasoning as it extracts SDOH risk factors and makes recommendations.
    Args:
        request (ClinicalNoteRequest): the text of a clinical note.
    Returns:
        SDOHResponse: the patient's SDOH risk factors and recommended interventions.
    """
    def event_generator():
        for event in graph.stream(note, stream_mode="updates"):
            for node_name, updated_state in event.items():
                if node_name == "extract_sdoh_risk_factors":
                    sdoh_agent_messages = {}
                    "identified these sdoh risk factors: housing instability, food insecurity"
                elif node_name == "map_to_z_codes":
                    sdoh_agent_messages = {
                
                    }
                elif node_name == "get_zipcode":
                    sdoh_agent_messages = {
                        
                    }
                elif node_name == "search_social_services":
                        sdoh_agent_messages = {
                        }
                elif node_name == "recommend_interventions":
                    sdoh_agent_messages = {
                        
                    }
        yield f"data: {json.dumps(sdoh_agent_messages)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"}
    )
    

@app.get("/")
async def health_check():
    """Tests to verify the API is running."""
    return {"status": "hello SDOH api is up", "time": time.time()}

# Main entry point
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"Health check called: SDOH API server on {host}:{port}")
    uvicorn.run(
        "api:app", 
        host=host, 
        port=port, 
        reload=False,  # Set to False in production
        log_level="info"
    )