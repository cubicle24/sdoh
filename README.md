# Social Determinants of Health (SDOH) Autonomous Agent

An AI-powered autonomous agent built with **LangGraph** and **LangChain** that analyzes clinical notes to extract SDOH risk factors, maps them to ICD-10 Z-codes, and recommends local social services interventions.

## 🎯 Project Overview

**Social Determinants of Health (SDOH)** are environmental, cultural, economic, and social factors responsible for **50-80% of health outcomes**. Common examples include:
- Food insecurity
- Housing instability  
- Financial hardship
- Transportation barriers
- Low health literacy
- Domestic violence

Despite their critical importance, SDOH factors are routinely overlooked in clinical care. This autonomous agent addresses this gap by:
1. **Automatically scanning** free-text clinical notes
2. **Extracting** SDOH risk factors using advanced NLP
3. **Mapping** findings to standardized ICD-10 Z-codes
4. **Recommending** local social services interventions

> **Note:** This agent is for educational and demonstration purposes and does not replace professional medical advice.

## 🏗️ Architecture

**Built with LangGraph State Machine:**
- **Multi-step autonomous workflow** with state management
- **Tool-calling capabilities** for social services lookup
- **Retry logic** and error handling
- **Audit trail** for healthcare compliance

**Key Components:**
- `sdoh_agent.py` - Core LangGraph state machine
- `social_services_tools.py` - Social services lookup tools (LLM decides using function calling)
- `api.py` - FastAPI REST endpoints
- `prompts/` - Carefully Engineered prompts for SDOH extraction

## 🚀 Features

### Core Functionality
- ✅ **SDOH Risk Factor Extraction** from clinical notes
- ✅ **ICD-10 Z-Code Mapping** for standardized documentation
- ✅ **Local Social Services Lookup** by zipcode and category
- ✅ **Autonomous Multi-Step Processing** with LangGraph
- ✅ **RESTful API** for healthcare system integration

### Healthcare Standards
- 📋 **ICD-10 Z-Code Compliance** for billing and quality metrics
- 🔍 **Structured Data Extraction** from unstructured clinical notes
- 📊 **Audit Trail** for healthcare compliance requirements

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. **Clone and setup environment:**
```bash
git clone <your-repo>
cd sdoh
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
cd src
pip install -r requirements.txt
```