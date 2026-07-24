# this file is going to be holding our reasoning engine
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import os

load_dotenv()

@tool # decorator turns this function into a LangChain tool that we can plug in
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

tools = [TavilySearch(max_results = 1), triple]

# utilize Function Calling
# initialize an llm and supply it with the tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)