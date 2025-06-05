from typing import Dict, List, Any
from langchain.tools import tool

@tool
def search_social_services(zipcode: str, service_category: str, radius: int) -> Dict[str, Any]:
    """
    Searches for social services within a given radius of a zipcode.
    Args:
        zipcode (str): The zipcode where you want to find social services for.
        service_category (str): The category of social service to search in one or more of the following choices: 
        Food, Housing, Employmeent, Education, Financial Assistance, Legal Assistance, Counseling
        radius (int): The radius in miles to search around the zipcode.
    Returns:
        Dict[str, Any]: A dictionary containing the search results.
    """
    #there are no free APIs for social services, so I am mocking this for the portfolio. If given access to a real API,
    # we'd simply make an API call
    neighborhood_resources = [
  {
    "name": "Issaquah Community Services",
    "address": "180 East Sunset Way, Issaquah, WA 98027",
    "phone": "425-837-3125",
    "website": "https://www.issaquahcommunityservices.org/",
    "services": [
      "Emergency rent assistance",
      "Utility assistance"
    ]
  },
  {
    "name": "Issaquah Food & Clothing Bank",
    "address": "179 1st Ave SE, Issaquah, WA 98027",
    "phone": "425-392-4123",
    "website": "https://issaquahfoodbank.org/",
    "services": [
      "Food assistance",
      "Clothing bank",
      "Hygiene and household necessities",
      "Financial assistance in partnership with Issaquah Community Services"
    ]
  },
  {
    "name": "Hopelink",
    "address": "11011 120th Ave NE, Kirkland, WA 98033",
    "phone": "425-869-6000",
    "website": "https://www.hopelink.org/",
    "services": [
      "Food banks",
      "Energy assistance",
      "Affordable housing",
      "Family development programs",
      "Transportation services",
      "Adult education programs"
    ]
  },
  {
    "name": "Salvation Army Eastside Corps",
    "address": "911 164th Ave NE, Bellevue, WA 98008",
    "phone": "425-452-7300",
    "website": "https://bellevue.salvationarmy.org/",
    "services": [
      "Emergency food assistance",
      "Utility assistance",
      "Domestic violence community advocacy program"
    ]
  }
]
    return neighborhood_resources

@tool
def get_patient_zipcode() -> str:
    """Obtains the zipcode of the patient. Use this tool when you need to find localized social services near the patient. 
    Args: None
    Returns:
        str: the zipcode of the patient.
    """

    zip = "98029"
    return zip


SOCIAL_SERVICES_TOOLS = [
    search_social_services,
    get_patient_zipcode
]