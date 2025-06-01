# Social Determinants of Health (SDOH) Autonomous Agent

An AI agent that scans clinical notes, extracts SDOH risk factors, maps to ICD10 Z-codes, and recommends local interventions.

Motivation: Social Determinants of Health (SDOH) are environmental, cultural, economic, and yes, social factors that are responsible for 50-80% of health outcomes.  Common examples are food insecurity, housing insecurity, financial hardship, lack of trasnportation, low health literacy, and domestic violence. It
is not hard to see why a patient who has one or more of these factors would be
less healthy than a patient who does not have these factors.

Despite their high importance, SDOH are routinely overlooked in clinical care. Documenting and directly addressing SDOH are also increasingly tied to reimubrsement and quality scores for providers and health systems.
An AI agent that continuously scans free-text clinical notes to identify patients who could benefit from SDOH interventions has potential to drive better health outcomes.

This agent is for educational purposes and does not replace professional medical advice.

## Features

- Extracts patient information from clinical notes
- Retrieves relevant clinical guidelines based on patient data
- Generates personalized screening recommendations
- Supports multiple LLM providers (Google Gemini, Groq/Llama)

## Setup

1. Install dependencies: pip install -r requirements.txt
2. Make a virtual environment: python3 -m venv guidelines_rag (you can use any name you want;
   here it is "guidelines_rag")
3. Activate the virtual environment you just created: source guidelines_rag/bin/activate
2. Set up environment variables:
    - `OPENAI_API_KEY`: Your OpenAI API key
    - `GEMINI_API_KEY`: Your Google Gemini API key
    - `GROQ_API_KEY`: Your Groq API key
3. Switch into the `src` directory: cd src
4. Run the main script: python main.py

## Usage

1. Provide a patient's clinical notes when prompted.
2. The system will extract patient information and retrieve relevant screening test guidelines 
based on the US Preventive Task Force recommendations.
3. The generated recommendations will be displayed. Screening tests that are based only on the 
local guidelines repository will be given first; additional ones based on a search of medical literature will also be displayed.

## API Usage

The system provides a REST API for integration with other applications:

1. Start the API server: `python api.py`
2. The API will be available at http://localhost:8000
3. API endpoints:
   - POST `/api/recommendations`: You give the system a clinical note, and screening recommendations will be returned
   - GET `/api/hello`: makes sure the API is running

API documentation is available at http://localhost:8000/docs when the server is running.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
