import torch
import sklearn

def calc_f1(logits, labels):
    """
    F1 Score 

    logits: (batch_size, 2, sequence_length), 2 is the sofmax dimension
    labels: (batch_size, sequence_length)
    """

    pred_vales = torch.max(logits, dim=1)[1] # (batch_size, sequence_len)

    flat_pred_values = torch.flatten(pred_vales)
    flat_labels = torch.flatten(labels)

    return sklearn.metrics.f1_score(
        y_true=flat_labels,
        y_pred=flat_pred_values
    )

