import time
import copy
import torch


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    device,
    num_epochs=25,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    early_stopper = EarlyStopper(patience=5)

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
                dataset_size = len(train_loader.dataset)
            else:
                model.eval()
                loader = val_loader
                dataset_size = len(val_loader.dataset)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize if train
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            if phase == "val":
                print(
                    f"Epoch {epoch + 1}/{num_epochs}: loss {epoch_loss:.4f}; acc: {epoch_acc:.4f}"
                )

            if phase == "train":
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

                # Deep copy the model if loss improved and accuracy is not lesser than 90% from best
                if epoch_loss < best_val_loss and epoch_acc >= (0.9 * best_val_acc):
                    best_val_loss = epoch_loss
                    best_val_acc = epoch_acc
                    best_epoch = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())

        if phase == "val" and early_stopper.early_stop(round(val_losses[-1], 2)):
            print(
                f"\nEarly stop triggered. No improvement in validation loss for {early_stopper.patience} epochs."
            )
            break

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Loss: {best_val_loss:.4f}; Epoch {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_losses, val_losses, best_epoch


class EarlyStopper:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
