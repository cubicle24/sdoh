from hmac import new
from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
import json
from typing import Dict, List, TypedDict, Callable, Any
import os, re
from pathlib import Path
from datetime import datetime
from pprint import pprint

from pydantic.v1.typing import AnyArgTCallable
from social_services_tools import SOCIAL_SERVICES_TOOLS
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool  # <-- Add this

class SDOHRiskFactor(TypedDict):
    """One patient social risk factor and its relevant properties"""
    #it's weird I know not to have the name nor intervention encapsulated here but it's not supposed to be an object
    present: bool
    reasoning: str
    z_code: List[str]

class AgentState(TypedDict):
    note : str
    #"housing: {present: bool, reasoning: str, z_code: List[str], intervention : str}"
    sdoh : Dict[str, SDOHRiskFactor]
    intervention: Dict[str, str]
    retry_count: int
    zipcode_tool_called : bool
    zipcode : str
    social_services : Dict[str, Any]
    audit_trail : Dict[str, Any]


def load_prompt(prompt_file_path: str, input_vars: List[str]) -> PromptTemplate:
    """Reads in a prompt from a text file, returns a Langchain PromptTemplate
    
    Args:
        prompt_file_path (str): path to the prompt file
        input_variables (List[str]): tells the template what variable placeholders to expect; e.g,  {note}
        variables (dict): the actual dictionary containing the values for those variables placeholders
    Returns:
        a Langchain formatted prompt template (which is a str)
    """
    prompt_string = Path(prompt_file_path).read_text()
    template = PromptTemplate(template=prompt_string, input_variables=input_vars)
    return template
    

def call_llm_with_tools(prompt: PromptTemplate, input_values: Dict, tools_list: List = None) -> Dict[str, Any]:
    """Calls the LLM (with or without tools) and returns the response (always a dict)"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    if not tools_list:
        response = (prompt | llm).invoke(input_values)
        return _parse_response(response.content)

    # Convert tools and bind to LLM
    converted_tools = [convert_to_openai_tool(tool) for tool in tools_list]
    llm_with_tools = llm.bind(tools=converted_tools)
    
    # LLM will decide which tools to use (contained in response)
    response = (prompt | llm_with_tools).invoke(input_values)

    #if wrong structure or no tools called 
    if not (hasattr(response, "tool_calls") and response.tool_calls):
        return _parse_response(response.content)

    chat_history = [HumanMessage(content=prompt.format(**input_values)),response]

    #there are tools to call, so execute them
    tool_responses = _execute_tools(response.tool_calls, tools_list)
    chat_history.extend(tool_responses)

    final_response = llm.invoke(chat_history)
    return _parse_response(final_response.content)


def _execute_tools(tool_calls: List, tools_list: List) -> List[ToolMessage]:
    """Execute tool calls and return ToolMessages"""
    # Create tool lookup dict for O(1) access
    tools_dict = {tool.name: tool for tool in tools_list}
    tool_messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call['name']
        tool_function = tools_dict.get(tool_name)
        
        if not tool_function:
            result = f"Error: Tool '{tool_name}' not found"
        else:
            try:
                result = str(tool_function.invoke(tool_call['args']))
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
        
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call['id'])
        )
    
    return tool_messages
  

def _parse_response(content: str) -> Dict[str, Any]:
    """Parses the response from the LLM into a JSON dict"""
    content = content.strip()
    if not content:
        return {"error" : "Empty LLM Response, can not parse"}
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}", "raw_content": content}


def extract_sdoh_risk_factors(state: AgentState) -> AgentState:
    """Extract social risk factors from a note"""
    clinical_note = state["note"]
    # prompt = load_prompt("../prompts/extract_sdoh_risk_factors.txt", ["clinical_note"])
    # prompt = load_prompt("../prompts/extract_sdoh_v2.txt", ["clinical_note"])
    prompt = load_prompt("../prompts/extract_sdoh_v3.txt", ["clinical_note"])
    risk_factors = call_llm_with_tools(prompt, {"clinical_note": clinical_note}, None)

    new_state = {**state, "sdoh": risk_factors}
    # print(f"State after Extracted SDOH risk factors: {new_state}")
    return new_state


def map_to_z_codes(state: AgentState) -> AgentState:
    """Map social risk factors to ICD10 Z codes"""
    sdoh_risk_factors = state["sdoh"]
    prompt = load_prompt("../prompts/extract_zcodes_v1.txt", ["sdoh_risk_factors"])
    z_codes = call_llm_with_tools(prompt, {"sdoh_risk_factors": sdoh_risk_factors}, None)

    new_state = {**state, "sdoh": z_codes}
    # print(f"State after Extracting z codes: {new_state}")
    return new_state

def get_zipcode(state: AgentState) -> AgentState:
    """Get the zipcode of the patient and adds it to the state"""
    try:
        prompt_text = "Get the patient's zipcode. You may use the available tools you have to determine it.  Return just the zipcode as a string and no extra text."
        prompt = PromptTemplate(template=prompt_text,input_variables=None)
        response = call_llm_with_tools(prompt, {}, SOCIAL_SERVICES_TOOLS)
        print(f"\nDEBUG: get_zipcode response: {response}\n")
        zipcode = None
        if len(response) == 5:
            zipcode = response
        return {**state, "zipcode": zipcode, "zipcode_tool_called": True}
    except Exception as e:
        print(f"Get zipcode node failed to get_zipcode: {e}")
        return {**state, "zipcode": None, "zipcode_tool_called": False}   


def search_social_services(state: AgentState) -> AgentState:
    """Search for social services based on the patient's zipcode"""
    try:
        prompt_text = """Search for local social services based on the patient's zipcode, which is {zipcode}. You may use the available tools to look for available services.
        Return just the list of social services as a JSON object and no extra text. Do not start the response with the word 'json'"""
        prompt = PromptTemplate(template=prompt_text,input_variables=["zipcode"])
        response = call_llm_with_tools(prompt, {"zipcode": state["zipcode"]}, SOCIAL_SERVICES_TOOLS)
        return {**state, "social_services": response['social_services'], "social_services_tool_called" : True}   
    except Exception as e:
        print(f"Error in search_social_services: {e}")
        return {**state, "social_services": None, "social_services_tool_called" : False}   


def zipcode_success_router(state: AgentState) -> str:
    """Determines if the zipcode was successfully extracted"""
    zipcode = state.get("zipcode")
    if zipcode:
        valid_zip = bool(re.match(r'^\d{5}(\d{4})?$', zipcode))
        if valid_zip:
            return "search_social_services"
        else:
            return "recommend_interventions"
    else:
        return "recommend_interventions"


def recommend_interventions(state: AgentState) -> AgentState:
    """Recommend interventions for each social risk factor"""
    sdoh_risk_factors = state.get("sdoh", {})
    social_services = state.get("social_services", {})

    prompt = load_prompt("../prompts/recommend_interventions_v3.txt", ["sdoh_risk_factors","social_services"])
    try:
        response = call_llm_with_tools(prompt, {"sdoh_risk_factors": sdoh_risk_factors,"social_services": social_services}, None)
        return {**state, "sdoh": response, "social_services" : social_services}   
    except Exception as e:
        print(f"Error in recommending interventions: {e}\n")
        return {**state, "errors": f"{response}: {e}"}


def end_processing(state: AgentState) -> AgentState:
    """Final node that marks the completion of SDOH processing"""
    print("Graph completed")
    return state


def audited_node_factory(node_func: Callable, node_name: str, ) -> Callable:
    """Wraps a node to add audit functionality"""
    def audited_node(state: AgentState) -> AgentState:
        """This actually makes the node with the auditing functionality"""
        result = node_func(state)
        if "audit_trail" not in state:
            new_state = {**state, "audit_trail": []}
        else:
            new_state = {**state} 
        audit_trail = new_state["audit_trail"]

        #deep comparison of values of two complex dicts
        def are_equal(val1, val2):
            if isinstance(val1, dict) and isinstance(val2, dict):
                return val1 == val2
            return val1 == val2

        changes = {
            "modified": {k: v for k, v in result.items() 
                        if k != "audit_trail" and k in result and k in state and not are_equal(result[k], state[k])},
            "added": {k: v for k, v in result.items() 
                    if k != "audit_trail" and k in result and k not in state},
            "removed": {k: state[k] for k in state 
                        if k != "audit_trail" and k in state and k not in result}
        }
        
        # Add what changed in this node 
        audit_trail.append({ 
            "step": node_name, 
            "timestamp": datetime.now().isoformat(), 
            "changes": changes
        }) 
        
        # Return updated state with traces 
        new_state =  {**result, "audit_trail": audit_trail}
        return new_state
    return audited_node #outer function is returning a ref to inner function to get called later


def build_agent(state: AgentState):
    """Build the SDOH agent"""
    graph = StateGraph(AgentState)
    graph.add_node("extract_sdoh_risk_factors", audited_node_factory(extract_sdoh_risk_factors,"extract_sdoh_risk_factors"))
    graph.add_node("map_to_z_codes", audited_node_factory(map_to_z_codes,"map_to_z_codes"))
    graph.add_node("get_zipcode", audited_node_factory(get_zipcode,"get_zipcode"))
    graph.add_node("search_social_services", audited_node_factory(search_social_services,"search_social_services"))
    graph.add_node("recommend_interventions", audited_node_factory(recommend_interventions,"recommend_interventions"))
    graph.add_node("end", audited_node_factory(end_processing,"end"))

    graph.add_edge("extract_sdoh_risk_factors", "map_to_z_codes")
    graph.add_edge("map_to_z_codes", "get_zipcode")
    graph.add_conditional_edges("get_zipcode",zipcode_success_router, 
    {
        "search_social_services": "search_social_services",
        "recommend_interventions": "recommend_interventions"
    })
    graph.add_edge("search_social_services", "recommend_interventions")
    graph.add_edge("recommend_interventions", "end")

    graph.set_entry_point("extract_sdoh_risk_factors")

    sdoh_agent =  graph.compile()
    return sdoh_agent


def run_agent(note: str, state: AgentState):
    """Runs the SDOH agent"""
    agent = build_agent(state)
    try:
        result = agent.invoke(state)
        return result
    except Exception as e:
        print(f"Error when running agent: {str(e)}")
        return {**state, "error": str(e)}


