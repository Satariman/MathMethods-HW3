import numpy as np
from sklearn import metrics


def missing_features(data, features):
    return [feature for feature in features if feature not in data.columns]


def add_predictions(model, data, features, threshold):
    result = data.copy()
    result['probability'] = model.predict_proba(result[features])[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


def classification_metrics(y_true, probability, threshold):
    y_pred = (np.asarray(probability) >= threshold).astype(int)
    return {
        'confusion_matrix': metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]),
        'precision': metrics.precision_score(y_true, y_pred, zero_division=0),
        'recall': metrics.recall_score(y_true, y_pred, zero_division=0),
        'f1': metrics.f1_score(y_true, y_pred, zero_division=0),
        'prediction': y_pred,
    }
