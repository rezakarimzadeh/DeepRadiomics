import yaml
import torch 
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def read_yaml_file(file_path: str) -> dict:
    """Read a YAML configuration file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_json(file_path: str, data: dict):
    """Save a dictionary to a JSON file."""
    import json
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def test_model(model, test_loader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                x, y, mask = batch['features'], batch['labels'], batch.get('pad_mask', None)
                logits = model(x.to(model.device), mask.to(model.device) if mask is not None else None)
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def test_model_graph(model, test_loader):
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch.to(model.device))
                probs = torch.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch.y.long().cpu().numpy())
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_classification_metrics(algorithm_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        'model': algorithm_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics