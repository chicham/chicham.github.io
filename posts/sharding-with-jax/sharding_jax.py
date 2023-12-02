# ---
# format:
#   html:
#     code-fold: true
#   ipynb: default
# title: Sharding with jax
# draft: true
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
# ---

# %% [markdown]
"""
# Pre-requisites: Parallelism for deep learning
"""

# %% [markdown]
"""
# Utils functions
"""
# %%
import os


import jax
from jax import random
import jax.numpy as jnp
import keras as K
from pathlib import Path

import tempfile
import shutil
# jax.distributed.initialize()

N_TRAIN = 8 * 2 ** (10 + 4)
N_EVAL = 8 * 2**10
BATCH_SIZE = 32

key = random.PRNGKey(0)

print("Download dataset")
(x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = jnp.expand_dims(x_train, -1)
x_test = jnp.expand_dims(x_test, -1)

input_shape = 28, 28, 1

print("Make model")
model = K.Sequential(
    [
        K.layers.Input(shape=input_shape),
        K.layers.Flatten(),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dropout(rate=0.5),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dropout(rate=0.5),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dropout(rate=0.5),
        K.layers.Dense(128, activation="relu"),
        K.layers.Dense(10, activation="softmax"),
    ]
)
model.summary()
loss_fn = K.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = K.optimizers.Adam(3e-4)
train_metric = K.metrics.CategoricalAccuracy()


def compute_loss(trainable_variables, non_trainable_variables, metrics_variables, x, y):
    y_pred, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, x
    )
    loss = loss_fn(y, y_pred)
    metrics_variables = train_metric.stateless_update_state(
        metrics_variables, y, y_pred
    )
    return loss, (non_trainable_variables, metrics_variables)


grad_fn = jax.value_and_grad(compute_loss, has_aux=True)


@jax.jit
def train_step(state, data):
    (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    ) = state
    x, y = data

    (loss, (non_trainable_variables, metric_variables)), grads = grad_fn(
        trainable_variables, non_trainable_variables, metric_variables, x, y
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )

    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
        metric_variables,
    )


logs_dir = Path("logs/").resolve()
logs_dir.mkdir(exist_ok=True)

trainable_variables = model.trainable_variables
non_trainable_variables = model.non_trainable_variables

optimizer.build(trainable_variables)
optimizer_variables = optimizer.variables
metrics_variables = train_metric.variables

state = (
    trainable_variables,
    non_trainable_variables,
    optimizer_variables,
    metrics_variables,
)

x_train = x_train[2 * BATCH_SIZE :]
y_train = y_train[2 * BATCH_SIZE :]


def run(state, x_train, y_train):
    acc_loss = 0
    for step in range(0, x_train.shape[0], BATCH_SIZE):
        data = x_train[step : step + BATCH_SIZE], y_train[step : step + BATCH_SIZE]
        with jax.profiler.TraceAnnotation("train_step"):
            loss, state = train_step(state, data)
            loss.block_until_ready()
            acc_loss += loss
        if step % 100 == 0:
            *_, metrics_variables = state
            for variable, value in zip(train_metric.variables, metrics_variables):
                variable.assign(value)
            print(f"Acc: {train_metric.result()}")
            print(f"Loss: {acc_loss / (step + 1)}")


# %% [markdown]
"""
# Experiments
## Baseline run on 1 device
"""

# %%
baseline_dir = logs_dir / "baseline"

if not baseline_dir.exists():
    with tempfile.TemporaryDirectory(prefix="sharding_jax_") as tmpdir:
        with jax.profiler.trace(tmpdir):
            run(state, x_train, y_train)
        shutil.move(tmpdir, baseline_dir)

# %% [markdown]
"""
## Replicated
"""
