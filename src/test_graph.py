from workflows.automl_graph import build_graph


graph = build_graph()

result = graph.invoke(
    {
        "dataset_path": "data/raw/Titanic-Dataset.csv"
    }
)

print(result)