import torch
import pytorch_lightning as pl
from torch import nn, optim, Tensor
from dataclasses import asdict
from typing import Callable, List
from training_pipeline.metric_calculators import (
    MetricCalculator,
)
from training_pipeline.metrics_containers import (
    MetricContainer,
)


class BottleneckBlock(nn.Module):
    """
    Inverted Bottleneck.
    Taken from "Scaling MLPs: A Tale of Inductive Bias" https://arxiv.org/pdf/2306.13575.pdf.
    The idea is to first expand the input to a wider hidden size, then apply a nonlinearity,
    and finally project back to the original dimension.
    """

    def __init__(self, thin_dim: int, wide_dim: int):
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


class Net(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_size_wide: int,
        hidden_size_thin: int,
    ):
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, hidden_size_thin)
        self.ln_input = nn.LayerNorm(normalized_shape=hidden_size_thin)

        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=hidden_size_thin) for _ in range(3)]
        )
        self.bottlenecks = nn.ModuleList(
            [
                BottleneckBlock(thin_dim=hidden_size_thin, wide_dim=hidden_size_wide)
                for _ in range(3)
            ]
        )

        self.ln_output = nn.LayerNorm(normalized_shape=hidden_size_thin)
        self.linear_output = nn.Linear(hidden_size_thin, out_features=output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_projection(x)
        x = self.ln_input(x)
        for layernorm, bottleneck in zip(self.layernorms, self.bottlenecks):
            x = x + bottleneck(layernorm(x))
        x = self.ln_output(x)
        x = self.linear_output(x)
        return x


class UniversalModel(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size_thin: int,
        hidden_size_wide: int,
        output_dim: int,
        learning_rate: float,
        metric_calculator: MetricCalculator,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        metrics_tracker: List[MetricContainer],
        enable_logger: bool,
    ) -> None:
        super().__init__()

        torch.manual_seed(1278)
        self.learning_rate = learning_rate
        self.net = Net(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_size_thin=hidden_size_thin,
            hidden_size_wide=hidden_size_wide,
        )
        self.metric_calculator = metric_calculator
        self.loss_fn = loss_fn
        self.metrics_tracker = metrics_tracker
        self.enable_logger = enable_logger

    def forward(self, x) -> Tensor:
        return self.net(x)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage):
        self.metric_calculator.to(self.device)

    def training_step(self, train_batch, batch_idx) -> Tensor:
        x, y = train_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=self.enable_logger,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        x, y = val_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, logger=self.enable_logger)

        self.metric_calculator.update(
            predictions=preds,
            targets=y.long(),
        )

    def on_validation_epoch_end(self) -> None:
        metric_container = self.metric_calculator.compute()
        for metric_name, metric_val in asdict(metric_container).items():
            self.log(
                metric_name,
                metric_val,
                prog_bar=True,
                logger=self.enable_logger,
            )

        if not self.trainer.sanity_checking:
            self.metrics_tracker.append(metric_container)
