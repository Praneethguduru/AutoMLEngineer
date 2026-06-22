from langgraph.graph import StateGraph, START, END

from workflows.state import AutoMLState
from agents.dataset_analyst import dataset_analyst_node
from agents.model_trainer import model_trainer_node
from agents.model_selector import model_selector_node
from agents.explainability_agent import explainability_node
from agents.report_generator import report_generator_node


def build_graph():

    graph = StateGraph(AutoMLState)

    graph.add_node("dataset_analyst",dataset_analyst_node)

    graph.add_node("model_trainer", model_trainer_node)

    graph.add_node("explainability", explainability_node)

    graph.add_node("model_selector", model_selector_node)

    graph.add_node("report_generator", report_generator_node)

    graph.add_edge(START,"dataset_analyst")

    graph.add_edge("dataset_analyst", "model_trainer")

    graph.add_edge("model_trainer", "model_selector")

    graph.add_edge("model_selector", "explainability")
    
    graph.add_edge("explainability", "report_generator")

    graph.add_edge("report_generator", END)

    return graph.compile()