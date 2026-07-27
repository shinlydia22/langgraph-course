from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

# implementing reasoning engine of our agent
    # it's gonna be a node where the agent can receive our message
    # agent will decide whether or not it can answer it
    # or if it needs to invoke a tool

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""

# defining our agent reasoning node
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    # make an LLM call with the user input
    # the LLM will do most of the heavy lifting becos it is binded with the tools
        # (the tools we binded it with in react.py?)
    # leverage function calling; invoke the LLM
    # role is system, put the system message in,
    # and also put in all of the messages that we have (via *state["messages"])
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    # returns a dictionary with key "messages" w/ value a list that contains response
    # langgraph then appends `response` to our state
    return {"messages": [response]}

# define tool node with the relevant tools
tool_node = ToolNode(tools)