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
import os
from pathlib import Path
from datetime import datetime
from pprint import pprint
from social_services_tools import SOCIAL_SERVICES_TOOLS


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

def load_prompt(prompt_file_path: str, input_vars: List[str]) -> str:
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

def call_llm(prompt: PromptTemplate, input_values: Dict) -> str:
    """Call the LLM with a Langchain PromptTemplate and return the response"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not found.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", 
    temperature=0.0, 
    google_api_key=google_api_key)
    chain = prompt | llm
    response = chain.invoke(input_values).content

    try:
        print(f"call LLM response: {response}")
        return JSON.loads(response)#should be valid JS
    except json.JSONDecodeError as e:
        raise ValueError(f"error calling LLM: {e}. Response: {response}")
    


def call_llm_with_tools(prompt: PromptTemplate, input_values: Dict, tools_list: List = None) -> str:
    """Call the LLM with a Langchain PromptTemplate and return the response. LLM decides whether or not to use tools"""

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not found.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", 
    temperature=0.0, 
    google_api_key=google_api_key)

    if tools_list:
        llm = llm.bind(tools=tools_list)

    chain = prompt | llm 
    response = chain.invoke(input_values)


    #response is no longer JSON, response.content is tool_calls
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"llm has decided to call these tools: {response.tool_calls}")
        #example tool_calls:     tool_calls=[
        # {
        #     'name': 'get_patient_zipcode',
        #     'args': {},
        #     'id': 'call_abc123'
        # }
    # ],
        #you still have to make it call the tools
        #store the responses of the tool calls
        tool_call_responses = []
        for tool_call in response.tool_calls:
            print(f"executing tool: {tool_call['name']} with args: {tool_call['args']}")

            #first find the tool
            tool_function = None #the function itself
            for tool in tools_list:
                if tool.name == tool_call['name']:
                    tool_function = tool
                    break#found the tool
            if tool_function:
                try:
                    response = tool_function.invoke(tool_call['args'])#execute it
                    tool_call_responses.append({"tool_call_id": tool_call['id'],"result" : response,'status': "success"})
                    print(f"tool response: {response}")
                except Exception as e:
                    print(f"error calling tool: {e}")
                    tool_call_responses.append({"tool_call_id": tool_call['id'],"result" : f"error with response when calling {tool_call['name']}: {e}",'status': "error"})
            else:
                print(f"a tool wanted to be called but not found in tools list--hallucination?: {tool_call['name']}")
                tool_call_responses.append({"tool_call_id": tool_call['id'],"result" : f"error calling tool: {tool_call['name']} not found in tools list","status": "not_found"})
        
        chat_history = []
        chat_history.append(HumanMessage(content=prompt.format(**input_values)))
        chat_history.append(response)#the initial response is an AIMessage

        #tool_results = [
    #     {
    #         'tool_call_id': 'call_abc123',  # Matches the original tool call
    #         'result': 'Weather is 72°F',     # What the tool returned
    #         'status': 'success'              # success/error/not_found
    #     },
    #     {
    #         'tool_call_id': 'call_def456',
    #         'result': 'Tool crashed: Connection timeout',
    #         'status': 'error'
    #     }
    # ]
    
        for tool_call_response in tool_call_responses:
            if not isinstance(tool_call_response, dict) or 'tool_call_id' not in tool_call_response:
                raise ValueError(f"tool call response with wrong structure. I don't make the rules. It Must be a dict and have a tool_call_id: {tool_call_response}")

            tool_call_response_str = str(tool_call_response['result'])#ToolMessages and LLMs want text strings
            chat_history.append(ToolMessage(content=tool_call_response_str, tool_call_id=tool_call_response['tool_call_id']))
        
        final_response = llm.invoke(chat_history).content
        
    else:#no tools called for, so just return the regular response (won't have tool_calls)
        print(f"llm has decided not to call any tools, just returning response: {response.content}")
        final_response = response.content
    try:
        return json.loads(final_response)
    except json.JSONDecodeError as e:
        print(f"couldn't parse {final_response} as json: {str(e)}")
        raise ValueError(f"LLM returned invalid JSON: {e}. Response: {response_content}")


def extract_sdoh_risk_factors(state: AgentState) -> AgentState:
    """Extract social risk factors from a note"""
    clinical_note = state["note"]
    # prompt = load_prompt("../prompts/extract_sdoh_risk_factors.txt", ["clinical_note"])
    # prompt = load_prompt("../prompts/extract_sdoh_v2.txt", ["clinical_note"])
    prompt = load_prompt("../prompts/extract_sdoh_v3.txt", ["clinical_note"])
    risk_factors = call_llm(prompt, {"clinical_note": clinical_note})
    # new_state = state.copy()
    # new_state["sdoh"] = risk_factors
    new_state = {**state, "sdoh": risk_factors}
    print(f"State after Extracted SDOH risk factors: {new_state}")
    return new_state


def map_to_z_codes(state: AgentState) -> AgentState:
    """Map social risk factors to ICD10 Z codes"""
    sdoh_risk_factors = state["sdoh"]
    prompt = load_prompt("../prompts/extract_zcodes_v1.txt", ["sdoh_risk_factors"])
    z_codes = call_llm(prompt, {"sdoh_risk_factors": sdoh_risk_factors})
    # new_state = state.copy()
    # new_state["z_codes"] = z_codes
    new_state = {**state, "sdoh": z_codes}
    print(f"State after Extracting z codes: {new_state}")
    return new_state

def recommend_interventions(state: AgentState) -> AgentState:
    """Recommend interventions for each social risk factor"""
    sdoh_risk_factors = state["sdoh"]
    prompt = load_prompt("../prompts/recommend_interventions_v1.txt", ["sdoh_risk_factors"])

    interventions = call_llm(prompt, {"sdoh_risk_factors": sdoh_risk_factors})
    # new_state["interventions"] = interventions
    print(f"sdoh dict after adding interventions: {interventions}")

    new_state = {**state, "sdoh": interventions}
    return new_state

def end_processing(state: AgentState) -> AgentState:
    """Final node that marks the completion of SDOH processing"""
    
    print("Graph completed")
    return state

def audited_node_factory(node_func: Callable, node_name: str, ) -> Callable:
    """Wraps a node to add audit functionality"""
    def audited_node(state: AgentState) -> AgentState:
        """This actually makes the node with the auditing functionality"""
        result = node_func(state)
        print(f"Executed node {node_name} with result: {result}")
        if "audit_trail" not in state:
            state= {**state, "audit_trail": []}
        audit_trail = state["audit_trail"]

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
    graph.add_node("recommend_interventions", audited_node_factory(recommend_interventions,"recommend_interventions"))
    graph.add_node("end", audited_node_factory(end_processing,"end"))

    graph.add_edge("extract_sdoh_risk_factors", "map_to_z_codes")
    graph.add_edge("map_to_z_codes", "recommend_interventions")
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


