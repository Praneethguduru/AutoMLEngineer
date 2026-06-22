from langgraph.graph import StateGraph, START, END

from workflows.state import AutoMLState
from agents.dataset_analyst import dataset_analyst_node


def build_graph():

    graph = StateGraph(AutoMLState)

    graph.add_node("dataset_analyst",dataset_analyst_node)

    graph.add_edge(START,"dataset_analyst")

    graph.add_edge("dataset_analyst",END)

    return graph.compile()