from transformers import Trainer

import contextlib
import functools
import glob
import math
import os
import random
import re
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import is_sagemaker_mp_enabled

class TrainerNoShuffle(Trainer):
    def __init__(
        self,
        model = None,
        args: TrainingArguments = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Callable[[], "PreTrainedModel"] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
        )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]: # disable shuffling
        return SequentialSampler(self.train_dataset)