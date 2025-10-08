# train_container/train.py
import numpy as np
import copy
import math
import mlflow # type: ignore
from logging import INFO
from logger import log
from typing import Union, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, shuffle_train=True):
    """
    Converts numpy data into PyTorch DataLoader objects.
    """
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Code Below Borrowed (but modified) from University of Thessalonika Federated Learning Paper
def get_optim(model, optim_name: str = "adam", lr: float = 1e-3):
    """Returns the specified optimizer for the model defined as a torch module."""
    if optim_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optim_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optim_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer: {optim_name} is not supported.")


def get_criterion(crit_name: str = "mse"):
    """Returns the specified loss function."""
    if crit_name == "mse":
        return torch.nn.MSELoss()
    elif crit_name == "l1":
        return torch.nn.L1Loss()
    else:
        raise NotImplementedError(f"Criterion: {crit_name} is not supported.")


def accumulate_metric(y_true: Union[np.ndarray, torch.Tensor],
                      y_pred: Union[np.ndarray, torch.Tensor]) -> Tuple[float, float, float, float, float]:
    """Computes standard regression metrics (MSE, RMSE, MAE, R2, NRMSE)."""

    if not isinstance(y_true, np.ndarray):
        y_true = y_true.cpu().numpy()
    if not isinstance(y_pred, np.ndarray):
        y_pred = y_pred.cpu().numpy()

    # Flatten if >2D (e.g., batch × seq × features)
    if y_true.ndim > 2:
        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1])

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    nrmse_list = []
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        for i in range(y_true.shape[1]):
            y_true_dim = y_true[:, i]
            rmse_dim = math.sqrt(mean_squared_error(y_true_dim, y_pred[:, i]))
            mean_true = np.mean(y_true_dim)
            if mean_true != 0:
                nrmse_list.append(rmse_dim / mean_true)

    nrmse = float(np.mean(nrmse_list)) if nrmse_list else 0.0

    return mse, rmse, mae, r2, nrmse


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, delta=0, trace=True, trace_func=log):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.trace = trace
        self.trace_func = trace_func
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._cache_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace:
                self.trace_func(INFO, f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._cache_checkpoint(val_loss, model)
            self.counter = 0

    def _cache_checkpoint(self, val_loss, model):
        """Caches model when validation loss decreases."""
        if self.trace:
            self.trace_func(INFO, f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Caching model ...")
        self.val_loss_min = val_loss
        self.best_model = copy.deepcopy(model)


def test(model, data, criterion, device="cuda", ss: bool = False) -> Tuple[float, float, float, float, float, float]:
    """Tests a trained model and returns metrics."""
    model.to(device)
    model.eval()
    
    y_true_list, y_pred_list = [], []
    total_loss = 0.
    
    with torch.no_grad():
        for x, y in data:
            x, y = x.to(device), y.to(device)

            if ss:
                # For testing, always use the model's own predictions (teacher_forcing_ratio = 0.0)
                y_pred = model(x, y, teacher_forcing_ratio=0.0)
                # Calculate loss against the endogenous part of the target tensor
                loss = criterion(y_pred, y[:, :, :model.n_endo_features])
                # trim exo features for comparison
                y = y[:, :, :model.n_endo_features]
            else:
                # Original logic for non-seq2seq models
                y_pred = model(x)
                loss = criterion(y_pred, y)

            total_loss += loss.item() * x.size(0)
            y_true_list.append(y.cpu())
            y_pred_list.append(y_pred.cpu())
            
    loss = total_loss / len(data.dataset)
    
    y_true = torch.cat(y_true_list).squeeze()
    y_pred = torch.cat(y_pred_list).squeeze()
    
    mse, rmse, mae, r2, nrmse = accumulate_metric(y_true, y_pred)
    
    return loss, mse, rmse, mae, r2, nrmse


def train(model: torch.nn.Module,
          train_loader,
          test_loader,
          device,
          epochs: int = 10,
          optimizer_type: str = "adam",
          lr: float = 1e-3,
          scheduled_learning: bool = False,
          scheduled_sampling: bool = False,
          ss_decay: float = 0.33,
          reg1: float = 0.,
          reg2: float = 0.,
          max_grad_norm: float = 0.,
          criterion: str = "mse",
          early_stopping: bool = True,
          patience: int = 50,
          log_per: int = 1,
          use_carbontracker: bool = False):
    """Trains a neural network defined as a torch module."""

    best_model, best_loss, best_epoch = None, float('inf'), -1
    
    optimizer = get_optim(model, optimizer_type, lr)
    loss_fn = get_criterion(criterion)


    # For Transformers
    if scheduled_learning:
        from transformers import get_linear_schedule_with_warmup

        num_training_steps = epochs * len(train_loader)
        warmup_steps = int(0.1 * num_training_steps)  # e.g., 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    
    monitor = EarlyStopping(patience, trace=log_per == 1) if early_stopping else None
    
    cb_tracker = None
    if use_carbontracker:
        try:
            from carbontracker.tracker import CarbonTracker # type: ignore
            cb_tracker = CarbonTracker(epochs=epochs, components="all", verbose=1)
        except ImportError:
            log(INFO, "CarbonTracker not found.")

    for epoch in range(epochs):
        if cb_tracker:
            cb_tracker.epoch_start()
        
        if scheduled_sampling:
            # Linearly decay the teacher forcing ratio from 1.0 to 0.0 over ss_decay_epochs
            teacher_forcing_ratio = max(0.0, 1.0 - (epoch / (epochs*ss_decay)))
            if (epoch + 1) % log_per == 0:
                log(INFO, f"Epoch {epoch + 1}: Using teacher_forcing_ratio: {teacher_forcing_ratio:.4f}")

        # Training Phase
        model.train().to(device)
        total_train_loss = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if scheduled_sampling:
                y_pred = model(x, y, teacher_forcing_ratio)
                # Ensure the model has 'n_endo_features' attribute
                loss = loss_fn(y_pred, y[:, :, :model.n_endo_features])
            else:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)

            # L1/L2 Regularization
            if reg1 > 0. or reg2 > 0.:
                l1_reg = 0.
                l2_reg = 0.
                for name, param in model.named_parameters():
                    if "bias" not in name:
                        l1_reg += torch.norm(param, 1)
                        l2_reg += torch.norm(param, 2)
                loss += reg1 * l1_reg + reg2 * l2_reg
                
            loss.backward()
            if max_grad_norm > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduled_learning:
                scheduler.step()
            total_train_loss += loss.item() * x.size(0)
            
        train_loss = total_train_loss / len(train_loader.dataset)

        # Evaluation Phase
        test_loss, test_mse, test_rmse, test_mae, test_r2, test_nrmse = test(model, test_loader, loss_fn, device, ss=scheduled_sampling)
        train_loss_eval, train_mse, train_rmse, train_mae, train_r2, train_nrmse = test(model, train_loader, loss_fn, device, ss=scheduled_sampling)

        # Logging
        if (epoch + 1) % log_per == 0:
            log(INFO, f"Epoch {epoch + 1} [Train]: loss {train_loss_eval:.4f}, mse: {train_mse:.4f}, rmse: {train_rmse:.4f}, mae: {train_mae:.4f}, r2: {train_r2:.4f}, nrmse: {train_nrmse:.4f}")
            log(INFO, f"Epoch {epoch + 1} [Test]: loss {test_loss:.4f}, mse: {test_mse:.4f}, rmse: {test_rmse:.4f}, mae: {test_mae:.4f}")

            mlflow.log_metrics({
                "train_loss": train_loss_eval,
                "test_loss": test_loss,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae
            }, step=epoch + 1)

        # Early Stopping Logic
        if early_stopping and monitor is not None:
            monitor(test_loss, model)
            if monitor.early_stop:
                log(INFO, "Early stopping triggered.")
                best_model = monitor.best_model
                best_loss = monitor.val_loss_min
                best_epoch = epoch + 1 - monitor.counter
                break
            else:
                best_model = monitor.best_model
                best_loss = monitor.val_loss_min
                best_epoch = epoch + 1 - monitor.counter
        else:
            if test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)
                best_epoch = epoch + 1
        
        if cb_tracker:
            cb_tracker.epoch_end()
            
    if cb_tracker:
        cb_tracker.stop()

    log(INFO, f"Best Loss: {best_loss:.4f} found at epoch: {best_epoch}")
    
    return best_model
