import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn


def plot_loss(
    train_losses,
    val_losses,
    epoch,
    title="",
):
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.lineplot(train_losses, label="Training Loss", ax=ax)
    sns.lineplot(val_losses, label="Validation Loss", ax=ax)
    plt.axvline(x=epoch, color="grey", linestyle="--", label="Best Epoch")

    ax.set(xlabel="Epochs", ylabel="Loss", title=title)
    plt.legend()
    sns.despine()


def evaluate_model(model, df_loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in df_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_probs, all_labels


def plot_confusion_matrix(conf_matrix, label_mapping, title=""):
    _, ax = plt.subplots(1, 1, figsize=(8, 7))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_mapping.keys(),
        yticklabels=label_mapping.keys(),
        ax=ax,
    )
    ax.set(xlabel="Predicted Label", ylabel="True Label", title=title)


def evaluate_model_on_test(model, criterion, test_loader, device):
    since = time.time()
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            test_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct_predictions.double() / total_samples
    average_test_loss = test_loss / total_samples

    time_elapsed = time.time() - since
    print(f"\nTest run in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    return all_preds, all_probs, all_labels, test_accuracy, average_test_loss
