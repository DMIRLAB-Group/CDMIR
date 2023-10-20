import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from causaldmir.graph import Graph
def graph_equal(true_graph: Graph, est_graph: Graph):
    assert type(true_graph) == type(est_graph)
    return (true_graph.to_numpy() == est_graph.to_numpy()).all()

def get_performance(fitted, real, threshold=0, drop_diag=True):
    if isinstance(fitted,Graph):
        fitted=fitted.to_numpy()
    if isinstance(real, Graph):
        real = real.to_numpy()
    fitted = np.abs(fitted)
    if drop_diag:
        fitted = fitted - np.diag(np.diag(fitted))
        real = real - np.diag(np.diag(real))

    f1 = f1_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    precision = precision_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    recall = recall_score(y_true=real.ravel(), y_pred=np.array(fitted.ravel() > threshold))
    temp_result = np.array((f1, precision, recall, threshold))
    result = pd.DataFrame(columns=['F1', "Precision", "Recall", "threshold"])
    result.loc[0] = temp_result
    return result