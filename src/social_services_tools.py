from typing import Dict


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
    neighborhood_resources = Dict()
    return neighborhood_resources

@tool
def get_patient_zipcode() -> str:
    """Obtains the zipcode of the patient. Use this tool when you need to find localized social services near the patient. 
    Args: None
    Returns:
        str: the zipcode of the patient.
    """

    zip = "98029"
    if len(zip) != 5 or not zip.isdigit():
        return "10001"
    return zip


SOCIAL_SERVICES_TOOLS = [
    search_social_services,
    get_patient_zipcode
]