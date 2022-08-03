import torch
import os
from torchmetrics import Accuracy, F1Score, Precision, Recall
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix


def get_metrics(y_pred, y_true):

    y_pred = torch.sigmoid(y_pred)

    metrics = {}
    acc = Accuracy(average="micro")
    f1 = F1Score(average="micro")
    precision = Precision(average="micro")
    recall = Recall(average="micro")

    metrics["acc"] = acc(y_pred, y_true)
    metrics["f1"] = f1(y_pred, y_true)
    metrics["precision"] = precision(y_pred, y_true)
    metrics["recall"] = recall(y_pred, y_true)

    y_pred_int = (y_pred >= 0.5).int()
    metrics["cm"] = confusion_matrix(y_true=y_true, y_pred=y_pred_int)

    return metrics


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters shape", "Total Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            param_shape = parameter.shape
            params = parameter.numel()
            table.add_row([name, param_shape, params])
            total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_loss(train_losses, val_losses, save_path):

    plt.plot(train_losses, "b", label="train_loss")
    plt.plot(val_losses, "r", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.title("Traing and Validation losses curve")
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.show()


def plot_confusion_matrix(
    cm, classes, save_path, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):

    plt.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(save_path, "cm.png"))
    plt.show()
