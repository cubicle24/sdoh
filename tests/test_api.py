import pytest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.api import app

class TestSDOHAPI:
    """Tests the SDOH endpoints"""

    def setup_method(self):
        """Setup the test client"""
        self.client = TestClient(app)
        self.sample_note = """Patient is a 55 year old male with a history of alcohol abuse and homelessness who presents for delirium tremens. 
        He was treated with an ativan drip and discharged 3 days later. He does not have a job and will live at the local shelter.  
        He has no known family and no emergency contact."""

    def test_health_check(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        response_data = response.json()
        expected = {
            "status": "hello SDOH api is up",
            "time": response_data["time"]  # Use the actual timestamp
        }
        assert response_data == expected

    def test_agent_simple_note(self):#no streaming
        """Tests the agent: does it parse sdoh risk factors and return interventions"""
        body = {
            "note": "Patient is a 68 year old male. He is homeless and lives in his car. He lost his job recently and relies on meals from churches. His car also broke down."
        }
        response = self.client.post("/sdoh/run_agent_sync", json=body)
        assert response.status_code == 200
        #LLMs have non-deterministic output, even with temp of 0.0, so we'll test structure of response instead
        data = response.json()
        print(f"here is response in test:###############:{data}")
        assert "sdoh" in data
        assert "audit_trail" in data
        assert len(data["audit_trail"]) > 0

    # def test_agent_no_sdoh_in_note(self):
    #     """Tests the agent: does it parse sdoh risk factors and return interventions"""
    #     body = {
    #         "note": "Joe would like a burger and fries.  When are you going to pick up the milk? Do not forget to do your homework.  Meet me at 7:00 PM for coffee if you have time."
    #     }
    #     response = self.client.post("/sdoh/run_agent_sync", json=body)
    #     #LLMs have non-deterministic output, even with temp of 0.0, so we'll test structure of response instead

    #     assert response.status_code == 200
    #     data = response.json()
    #     assert "sdoh" in data
    #     assert "audit_trail" in data
    #     assert len(data["audit_trail"]) > 0

    def test_agent_simple_note_streaming(self):
        body = {
            "note": "Patient is a 68 year old male. He is homeless and lives in his car. He lost his job recently and relies on meals from churches. His car also broke down."
        }
        response = self.client.post("/sdoh/run_agent", json=body)
        assert response.status_code == 200
        
        # Parse SSE format
        content = response.content.decode('utf-8')
        lines = content.strip().split('\n')
        
        # Extract JSON from SSE format
        json_data = []
        for line in lines:
            if line.startswith('data: '):
                json_str = line[6:]  # Remove 'data: ' prefix
                if json_str.strip():  # Skip empty data
                    json_data.append(json.loads(json_str))
        
        # Test that we got some data
        assert len(json_data) > 0
        
        # Test structure of at least one message
        for data in json_data:
            if 'step' in data:
                assert isinstance(data['step'], str)
    
            #     "example": {
            # "sdoh": {
            #     "housing_instability": {"present": False, "reasoning": "No mention of housing issues.", "z_code": [], "interventions": []},
            #     "food_insecurity": {"present": False, "reasoning": "No mention of food insecurity.", "z_code": [], "interventions": []},
            #     "lack_of_transportation": {"present": False, "reasoning": "No mention of transportation issues.", "z_code": [], "interventions": []},
            #     "financial_hardship": {"present": False, "reasoning": "No mention of financial issues.", "z_code": [], "interventions": []},
            #     "domestic_violence": {"present": True, "reasoning": "Patient reports being hit by her husband.", "z_code": ["Z62.81"], "interventions": ["Contact the Domestic Violence Community Advocacy Program at Salvation Army Eastside Corps for support and resources.", "Consider counseling or therapy services specializing in domestic violence.", "Explore local shelters or safe houses if immediate safety is a concern."]},
            #     "language_barriers": {"present": False, "reasoning": "No mention of language barriers.", "z_code": [], "interventions": []},
            #     "low_health_literacy": {"present": True, "reasoning": "Patient unable to specify therapy learnings.", "z_code": ["Z55.9"], "interventions": ["Provide educational materials in simpler language.", "Offer one-on-one health education sessions to improve understanding of health conditions and treatments.", "Utilize teach-back methods to ensure comprehension of health information."]}
            # },
            # "audit_trail": [
            #     {
            #     "step": "extract_sdoh_risk_factors",
            #     "timestamp": "2025-06-05T19:11:23.099026",
            #     "changes": {
            #         "modified": {
            #         "sdoh": {
            #             "housing_instability": {"present": False, "reasoning": "No mention of housing issues."},
            #             "food_insecurity": {"present": False, "reasoning": "No mention of food insecurity."},
            #             "lack_of_transportation": {"present": False, "reasoning": "No mention of transportation issues."},
            #             "financial_hardship": {"present": False, "reasoning": "No mention of financial issues."},
            #             "domestic_violence": {"present": True, "reasoning": "Patient reports hitting her husband."},
            #             "language_barriers": {"present": False, "reasoning": "No mention of language barriers."},
            #             "low_health_literacy": {"present": True, "reasoning": "Patient unable to specify therapy learnings."}
            #         }
            #         },
            #         "added": {},
            #         "removed": {}
            #     }
            #     }    
            # ]
            # }