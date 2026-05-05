from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def classification_metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else None
    return {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1, "auc": auc}
