import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..utils.factory import (create_criterion, create_optimizer,
                             create_scheduler)
from ..utils.metric import AverageMeter, MetricEvaluator

_logger = logging.getLogger("train")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: DictConfig,
        train_loader: Optional[DataLoader] = None,
        valid_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Initialize the Trainer class.

        Args:
            model (torch.nn.Module): The neural network model to train.
            cfg (DictConfig): Configuration object containing training settings.
            train_loader (Optional[DataLoader]): DataLoader for training data.
            valid_loader (Optional[DataLoader]): DataLoader for validation data.
            test_loader (Optional[DataLoader]): DataLoader for test data.
        """
        self.model = model.to(cfg.device)
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.criterion = create_criterion(cfg.loss)
        self.optimizer = create_optimizer(self.model, cfg.optimizer)
        self.scheduler = create_scheduler(self.optimizer, cfg.scheduler)

        self.logging_interval = cfg.logger.interval
        self.wandb = cfg.logger.use_wandb

        self.metric_evaluator = MetricEvaluator(num_classes=model.num_classes)

        self.save_path = cfg.model_save_path
        os.makedirs(self.save_path, exist_ok=True)

    def fit(self) -> None:
        """
        Train the model across epochs and validate after each epoch.
        Saves the best and last model checkpoints and logs metrics.
        """
        best_acc = 0.0
        step = 0
        history = {}
        
        for epoch in range(self.cfg.max_epochs):
            _logger.info(f"\nEpoch: {epoch+1}/{self.cfg.max_epochs}")
            train_metrics = self._run_epoch(self.train_loader, mode="train")
            val_metrics = self._run_epoch(self.valid_loader, mode="val")

            metrics = OrderedDict(lr=self.optimizer.param_groups[0]["lr"])
            metrics.update([(f"train_{k}", v) for k, v in train_metrics.items()])
            metrics.update([(f"val_{k}", v) for k, v in val_metrics.items()])
            history[epoch] = metrics
            
            step += 1
            if self.wandb:
                wandb.log(metrics, step=step)

            if self.scheduler:
                self.scheduler.step()

            # model checkpoint
            if val_metrics["acc"] > best_acc:
                state = {"best_epoch": epoch, "best_acc": val_metrics["acc"]}
                self._save_model("best_model.pt")
                self._save_json(state, "best_results.json")
                best_acc = val_metrics["acc"]
                _logger.info(
                    "Best Accuracy {0:.3%} to {1:.3%}".format(
                        best_acc, val_metrics["acc"]
                    )
                )

        _logger.info(
            "Best Metric: {0:.3%} (epoch {1:})".format(
                state["best_acc"], state["best_epoch"]
            )
        )
        self._save_model("last_model.pt")
        self._save_json(history, "history.json")

    def test(self, save_name: Optional[str] = None) -> None:
        """
        Evaluate the trained model on the test set and save test results.

        Args:
            save_name (Optional[str]): Prefix for saved result files. Defaults to 'test_results'.
        """
        if save_name is None:
            save_name = 'test_results'
            print(f"{save_name}-per_class.json")
            return
        if self.test_loader is None:
            _logger.warning("Test loader is None. Skipping test.")
            return

        test_metrics = self._run_epoch(self.test_loader, mode="test", return_per_class=True)

        self._save_json(test_metrics.get("per_class", {}), f"{save_name}-per_class.json")
        test_metrics.pop("per_class", None)
        self._save_json(test_metrics, f"{save_name}.json")
        
    def _run_epoch(
        self, dataloader: DataLoader, mode: str, return_per_class: bool = False
    ) -> Dict[str, float]:
        """
        Run a single epoch of training or evaluation.

        Args:
            dataloader (DataLoader): DataLoader for the dataset.
            mode (str): Either "train", "val", or "test".
            return_per_class (bool): If True, return per-class metrics.

        Returns:
            Dict[str, float]: Dictionary containing performance metrics.
        """
        acc_m, loss_m = AverageMeter(), AverageMeter()
        total_score, total_pred, total_true = [], [], []
        
        if mode == 'train':
            training = True
            self.model.train()
        else:
            training = False
            self.model.eval()
        
        end = time.time()
        
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            acc = self.metric_evaluator.accuracy(outputs, targets)
            acc_m.update(acc, n=targets.size(0))
            loss_m.update(loss.item())
            
            total_score.extend(outputs.detach().cpu().tolist())
            total_pred.extend(outputs.argmax(dim=1).cpu().tolist())
            total_true.extend(targets.cpu().tolist())
            
            if training and idx % self.logging_interval == 0 and idx != 0:
                _logger.info(
                    f"TRAIN [{idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss_m.val:.4f} ({loss_m.avg:.4f}) "
                    f"Acc: {acc_m.avg:.3%} "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.3e} "
                )
        
        metrics = self.metric_evaluator.calc_metrics(
            y_true=total_true,
            y_score=total_score,
            y_pred=total_pred,
            return_per_class=return_per_class,
        )
        metrics.update({"acc": acc_m.avg, "loss": loss_m.avg})
        self._log_metrics(mode, metrics)

        return metrics
    
    def _log_metrics(self, mode: str, metrics: Dict[str, float]) -> None:
        """
        Log evaluation metrics to the console.

        Args:
            mode (str): Mode string indicating the phase (train/val/test).
            metrics (Dict[str, float]): Dictionary of evaluation metrics.
        """
        _logger.info(
            f"\n{mode.upper()}: Loss: {metrics['loss']:.3f} | "
            f"Acc: {metrics['acc'] * 100:.3f}% | "
            f"AUROC: {metrics['auroc'] * 100:.3f}% | "
            f"F1-Score: {metrics['f1'] * 100:.3f}% | "
            f"Recall: {metrics['recall'] * 100:.3f}% | "
            f"Precision: {metrics['precision'] * 100:.3f}%"
        )

    def _save_json(self, obj: dict, filename: str) -> None:
        """
        Save a dictionary object to a JSON file.

        Args:
            obj (dict): Dictionary object to save.
            filename (str): Name of the file to write.
        """
        with open(os.path.join(self.save_path, filename), "w") as f:
            json.dump(obj, f, indent=4)

    def _save_model(self, filename: str) -> None:
        """
        Save the current model's state_dict to a file.

        Args:
            filename (str): Name of the file to save the model weights.
        """
        torch.save(self.model.state_dict(), os.path.join(self.save_path, filename))