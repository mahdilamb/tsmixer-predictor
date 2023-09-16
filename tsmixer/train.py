# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train and evaluate models for time series forecasting."""

import glob
import logging
import os
import time
from typing import Literal

import joblib
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)


def train(
    *,
    seed: int = 0,
    model_name: Literal["tsmixer", "tsmixer_rev_in", "cnn", "full_linear"] = "tsmixer",
    data: Literal[
        "electricity",
        "exchange_rate",
        "national_illness",
        "traffic",
        "weather",
        "ETTm1",
        "ETTm2",
        "ETTh1",
        "ETTh2",
    ] = "weather",
    feature_type: Literal["S", "M", "MS"] = "M",
    target: str = "OT",
    checkpoint_dir: str = "./checkpoints/",
    delete_checkpoint: bool = False,
    seq_len: int = 336,
    pred_len: int = 96,
    n_block: int = 2,
    ff_dim: int = 2048,
    dropout: float = 0.05,
    norm_type: Literal["L", "B"] = "B",
    activation: Literal["relu", "gelu"] = "relu",
    kernel_size: int = 4,
    train_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.0001,
    patience: int = 5,
    result_path: str = "result.csv",
    log_dir: str = "logs",
    out_dir: str = ".",
):
    """Train a model.

    Parameters
    ----------
    seed : int, optional
        random seed, by default 0
    model_name : Literal['tsmixer', 'tsmixer_rev_in', 'cnn', 'full_linear'], optional
        model name, by default "tsmixer"
    data : Literal[ 'electricity', 'exchange_rate', 'national_illness', 'traffic', 'weather', 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2', ], optional
        data name, by default "weather"
    feature_type : Literal['S', 'M', 'MS'], optional
        forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate, by default "M"
    target : str
        target feature in S or MS task default="OT"
    checkpoint_dir : str, optional
        location of model checkpoints, by default './checkpoints/'
    delete_checkpoint : bool, optional
        delete checkpoints after the experiment, by default False
    seq_len : int, optional
        input sequence length, by default 336
    pred_len : int, optional
        prediction sequence length, by default 96
    n_block : int, optional
        number of block for deep architecture, by default 2
    ff_dim : int, optional
        fully-connected feature dimension, by default 2048
    dropout : float, optional
        dropout rate, by default .05
    norm_type : Literal['L','B'], optional
        LayerNorm or BatchNorm, by default "B"
    activation : Literal['relu', 'gelu'], optional
        Activation function, by default "relu"
    kernel_size : int, optional
        kernel size for CNN, by default 4
    temporal_dim : int, optional
        temporal feature dimension, by default 16
    hidden_dim : int, optional
        hidden feature dimension, by default 64
    num_workers : int, optional
        data loader num workers, by default 10
    train_epochs : int, optional
        train epochs, by default 100
    batch_size : int, optional
        batch size of input dat, by default 32
    learning_rate : float, optional
        optimizer learning rate, by default 0.0001
    patience : int, optional
        number of epochs to early stop, by default 5
    result_path : str, optional
        path to save result, by default "result.csv"
    log_dir : str, optional
        directory for tensorboard output, by default "logs"
    """

    checkpoint_dir = os.path.join(out_dir, checkpoint_dir)
    result_path = os.path.join(out_dir, result_path)
    log_dir = os.path.join(out_dir, log_dir)

    import tensorflow as tf

    from tsmixer.models import tsmixer_basic
    from tsmixer.utils import data_loaders

    if "tsmixer" in model_name:
        exp_id = f"{data}_{feature_type}_{model_name}_sl{seq_len}_pl{pred_len}_lr{learning_rate}_nt{norm_type}_{activation}_nb{n_block}_dp{dropout}_fd{ff_dim}"
    elif model_name == "full_linear":
        exp_id = f"{data}_{feature_type}_{model_name}_sl{seq_len}_pl{pred_len}_lr{learning_rate}"
    elif model_name == "cnn":
        exp_id = f"{data}_{feature_type}_{model_name}_sl{seq_len}_pl{pred_len}_lr{learning_rate}_ks{kernel_size}"
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    data_loader = data_loaders.TSFDataLoader(
        data,
        batch_size,
        seq_len,
        pred_len,
        feature_type,
        target,
    )
    joblib.dump(data_loader.scaler, os.path.join(out_dir, "scaler.joblib"))
    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()
    model: tf.keras.Model
    # train model
    if "tsmixer" in model_name:
        build_model = getattr(tsmixer_basic, model_name).build_model
        model = build_model(
            input_shape=(seq_len, data_loader.n_feature),
            pred_len=pred_len,
            norm_type=norm_type,
            activation=activation,
            dropout=dropout,
            n_block=n_block,
            ff_dim=ff_dim,
            target_slice=data_loader.target_slice,
        )
    elif model_name == "full_linear":
        model = tsmixer_basic.full_linear.Model(
            n_channel=data_loader.n_feature,
            pred_len=pred_len,
        )
    elif model_name == "cnn":
        model = tsmixer_basic.cnn.Model(
            n_channel=data_loader.n_feature,
            pred_len=pred_len,
            kernel_size=kernel_size,
        )
    else:
        raise ValueError(f"Model not supported: {model_name}")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    checkpoint_path = os.path.join(checkpoint_dir, f"{exp_id}_best")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience
    )

    try:
        first_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    except ModuleNotFoundError:
        start_training_time = time.time()
        first_callback = tf.keras.callbacks.LambdaCallback(
            on_train_end=lambda: print(
                f"Training finished in {time.time() - start_training_time} secconds"
            )
        )
    callbacks = [first_callback, checkpoint_callback, early_stop_callback]
    history = model.fit(
        train_data,
        epochs=train_epochs,
        validation_data=val_data,
        callbacks=callbacks,
    )

    # evaluate best model
    best_epoch = np.argmin(history.history["val_loss"])
    model.load_weights(checkpoint_path)
    test_result = model.evaluate(test_data)
    if delete_checkpoint:
        for f in glob.glob(checkpoint_path + "*"):
            os.remove(f)

    # save result to csv
    run_data = {
        "data": [data],
        "model": [model_name],
        "seq_len": [seq_len],
        "pred_len": [pred_len],
        "lr": [learning_rate],
        "mse": [test_result[0]],
        "mae": [test_result[1]],
        "val_mse": [history.history["val_loss"][best_epoch]],
        "val_mae": [history.history["val_mae"][best_epoch]],
        "train_mse": [history.history["loss"][best_epoch]],
        "train_mae": [history.history["mae"][best_epoch]],
        "norm_type": norm_type,
        "activation": activation,
        "n_block": n_block,
        "dropout": dropout,
    }
    if "tsmixer" in model_name:
        run_data["ff_dim"] = ff_dim

    df = pd.DataFrame(run_data)
    if os.path.exists(result_path):
        df.to_csv(result_path, mode="a", index=False, header=False)
    else:
        df.to_csv(result_path, mode="w", index=False, header=True)
