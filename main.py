from dotenv import load_dotenv
from langchain_core.messages import HumanMessage # the input we start our graph to run with
from langgraph.graph import END, MessagesState, StateGraph

from nodes import run_agent_reasoning, tool_node

load_dotenv()

# defining the graph

# start by defining some constants! (for cleaner and more readable code)
AGENT_REASON = "agent_reasoning"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    # if last message is a tool call, go to act node
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON) # add edge btw start node and agent_reason node
flow.add_node(ACT, tool_node)

# should_continue is the functionality that decides which node to go to next
# third arg is a dictionary; maps potential outputs of should_continue to nodes
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path = "flow.png")

def main():
    print("Hello ReAct LangGraph with Function Calling!")
    # okie so now we invoke the graph! (we can cos it is runnable heheh)
    res = app.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo? List it and then triple it.")]})
    # it should invoke Tavily search tool and then the triple tool
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
