import torch
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from structurednets.asset_helpers import load_features

def get_accuracy(y_true: torch.tensor, y_pred: torch.tensor):
    return (torch.sum(y_pred.argmax(axis=1) == y_true) / len(y_true)).cpu().detach().numpy()

def get_loss(y_true: torch.tensor, y_pred: torch.tensor, loss_function_class=torch.nn.CrossEntropyLoss):
    loss_function = loss_function_class()
    return loss_function(y_pred, target=y_true).cpu().detach().numpy()

def get_train_data(features_path: str):
    X, y, _ = load_features(features_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X_train, X_val, y_train, y_val

def get_batch(X: np.ndarray, y: np.ndarray, batch_size: int, batch_i: int):
    X_batch = X[int(batch_i*batch_size):min(int((batch_i+1)*batch_size), X.shape[0])]
    y_batch = y[int(batch_i*batch_size):min(int((batch_i+1)*batch_size), y.shape[0])]
    X_batch_t, y_batch_t = map(torch.tensor, (X_batch, y_batch))
    return X_batch_t, y_batch_t

def get_full_batch(X: np.ndarray, y: np.ndarray):
    return get_batch(X=X, y=y, batch_size=X.shape[0], batch_i=0)

def transform_feature_dtypes(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray):
    return X_train.astype(np.float32), X_val.astype(np.float32), y_train.astype(np.int64), y_val.astype(np.int64)

def train_with_features(model: torch.nn.Module, features_path: str, patience=10, batch_size=1000, verbose=False, lr=1e-6, restore_best_model=True, loss_function_class=torch.nn.CrossEntropyLoss):
    X_train, X_val, y_train, y_val = get_train_data(features_path)
    X_train, X_val, y_train, y_val = transform_feature_dtypes(X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    return train(
        model=model, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
        patience=patience, batch_size=batch_size, verbose=verbose, lr=lr, 
        restore_best_model=restore_best_model, loss_function_class=loss_function_class,
    )

def get_loss_and_accuracy_for_model(model: torch.nn.Module, X_t: torch.tensor, y_t: torch.tensor, loss_function_class=torch.nn.CrossEntropyLoss):
    y_pred = model(X_t)
    loss = get_loss(y_t, y_pred, loss_function_class=loss_function_class)
    if len(y_t.shape) == 2 and y_t.shape[1] > 1:
        accuracy = get_accuracy(y_t.argmax(axis=1), y_pred)
    else:
        accuracy = get_accuracy(y_t, y_pred)
    return loss, accuracy

def train_with_decreasing_lr(model: torch.nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, patience=10, batch_size=1000, verbose=False, loss_function_class=torch.nn.CrossEntropyLoss, min_patience_improvement=1e-10, optimizer_class=torch.optim.SGD):
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # NOTE: We don't use an automatic way for scheduling the learning rate (such as for example torch.optim.ReduceLROnPlateau, because then we can restore the best model between all optimization steps)
    lr = 1e-1

    trained_model = model
    start_training_losses = []
    start_training_accuracies = []
    start_val_losses = []
    start_val_accuracies = []
    train_loss_histories = []
    train_accuracy_histories = []
    val_loss_histories = []
    val_accuracy_histories = []
    for _ in range(5):
        trained_model, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, patience=patience, batch_size=batch_size, verbose=verbose, lr=lr, restore_best_model=True, loss_function_class=loss_function_class, min_patience_improvement=min_patience_improvement, optimizer_class=optimizer_class)
        
        start_training_losses.append(start_train_loss)
        start_training_accuracies.append(start_train_accuracy)
        start_val_losses.append(start_val_loss)
        start_val_accuracies.append(start_val_accuracy)
        train_loss_histories.append(train_loss_history)
        train_accuracy_histories.append(train_accuracy_history)
        val_loss_histories.append(val_loss_history)
        val_accuracy_histories.append(val_accuracy_history)
        
        lr *= 1e-1

    return trained_model, start_training_losses, start_training_accuracies, start_val_losses, start_val_accuracies, train_loss_histories, train_accuracy_histories, val_loss_histories, val_accuracy_histories

def train(model: torch.nn.Module, X_train: np.ndarray, y_train: np.ndarray, X_val=None, y_val=None, patience=10, batch_size=1000, verbose=False, lr=1e-6, restore_best_model=True, loss_function_class=torch.nn.CrossEntropyLoss, min_patience_improvement=1e-10, optimizer_class=torch.optim.Adam):
    if X_val is None or y_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    X_train_t, y_train_t = get_full_batch(X_train, y_train)
    X_val_t, y_val_t = get_full_batch(X_val, y_val)

    optimizer = optimizer_class(model.parameters(), lr=lr)
    
    loss_function = loss_function_class()
    nb_batches_per_epoch = np.ceil(X_train.shape[0] / batch_size).astype("int")

    start_train_loss, start_train_accuracy = get_loss_and_accuracy_for_model(model=model, X_t=X_train_t, y_t=y_train_t, loss_function_class=loss_function_class)
    start_val_loss, start_val_accuracy = get_loss_and_accuracy_for_model(model=model, X_t=X_val_t, y_t=y_val_t, loss_function_class=loss_function_class)

    train_loss_history = []
    train_accuracy_history = []
    val_loss_history = []
    val_accuracy_history = []

    best_val_loss = start_val_loss
    best_model = pickle.loads(pickle.dumps(model))

    continue_training = True
    epoch = 1
    while continue_training:
        X_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

        for batch_i in range(nb_batches_per_epoch):
            X_batch_t, y_batch_t = get_batch(X_train_shuffled, y_train_shuffled, batch_size=batch_size, batch_i=batch_i)
            outputs_train = model(X_batch_t)
            loss_train = loss_function(outputs_train, target=y_batch_t)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
        
        train_loss, train_accuracy = get_loss_and_accuracy_for_model(model=model, X_t=X_train_t, y_t=y_train_t, loss_function_class=loss_function_class)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        val_loss, val_accuracy = get_loss_and_accuracy_for_model(model=model, X_t=X_val_t, y_t=y_val_t, loss_function_class=loss_function_class)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

        if verbose:
            print("--- Epoch " + str(epoch) + " ---")
            print("Train Acc: " + str(train_accuracy_history[-1]))
            print("Train Loss: " + str(train_loss_history[-1]))
            print("Val Acc: " + str(val_accuracy_history[-1]))
            print("Val Loss: " + str(val_loss_history[-1]))

        if len(val_loss_history) > 2*patience \
            and np.min(val_loss_history[-patience:]) >= np.min(val_loss_history[:-patience]) - min_patience_improvement:
            continue_training = False

        if val_loss_history[-1] < best_val_loss:
            best_val_loss = val_loss_history[-1]
            best_model = pickle.loads(pickle.dumps(model))

        epoch += 1
    
    if restore_best_model:
        model_to_return = best_model
    else: 
        model_to_return = model

    return model_to_return, start_train_loss, start_train_accuracy, start_val_loss, start_val_accuracy, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history