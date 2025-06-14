from pickle import FALSE
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


    @pytest.mark.skip(reason="working")
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


    @pytest.mark.skip(reason="basic validation working, slow expensive test")
    def test_agent_simple_note(self):#no streaming
        """Tests the agent: does it parse sdoh risk factors and return interventions"""
        body = {
            "note": "Patient is a 68 year old male. He is homeless and lives in his car. He lost his job recently and relies on meals from churches. His car also broke down."
        }
        response = self.client.post("/sdoh/run_agent_sync", json=body)
        assert response.status_code == 200
        #LLMs have non-deterministic output, even with temp of 0.0, so we'll test structure of response instead
        data = response.json()
        assert "sdoh" in data
        assert "audit_trail" in data
        assert len(data["audit_trail"]) > 0


    @pytest.mark.skip(reason="no sdoh validation working")
    def test_agent_no_sdoh_in_note(self):
        """This is a nonsensical note with nothing to do with SDOH. It should not return any risk factors present but still try anyway"""
        body = {
            "note": "Joe would like a burger and fries.  When are you going to pick up the milk? Do not forget to do your homework.  Meet me at 7:00 PM for coffee if you have time."
        }
        response = self.client.post("/sdoh/run_agent_sync", json=body)
        #LLMs have non-deterministic output, even with temp of 0.0, so we'll test structure of response instead

        assert response.status_code == 200
        data = response.json()
        assert "sdoh" in data
        assert "housing_instability" in data["sdoh"]
        assert "food_insecurity" in data["sdoh"]
        assert "lack_of_transportation" in data["sdoh"]
        assert "financial_hardship" in data["sdoh"]
        assert "domestic_violence" in data["sdoh"]
        assert "language_barriers" in data["sdoh"]
        assert "low_health_literacy" in data["sdoh"]

        assert data["sdoh"]["housing_instability"]["present"] == False
        assert data["sdoh"]["food_insecurity"]["present"] == False
        assert data["sdoh"]["lack_of_transportation"]["present"] == False
        assert data["sdoh"]["financial_hardship"]["present"] == False
        assert data["sdoh"]["domestic_violence"]["present"] == False
        assert data["sdoh"]["language_barriers"]["present"] == False
        assert data["sdoh"]["low_health_literacy"]["present"] == False

        assert "audit_trail" in data
        assert len(data["audit_trail"]) > 0


    def test_agent_simple_note_streaming(self):
        body = {
            "note": "Patient is a 68 year old male. He is homeless and lives in his car. He lost his job recently and relies on meals from churches. His car also broke down."
        }
        response = self.client.post("/sdoh/run_agent", json=body)
        assert response.status_code == 200
        
        # Parse SSE format
        content = response.content.decode('utf-8')
        lines = content.strip().split('\n')

        print(f"content in streaming test: {content}")
        
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
    