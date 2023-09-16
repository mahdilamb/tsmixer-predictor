import os
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt

from tsmixer.models import tsmixer_basic
from tsmixer.utils import data_loaders


def forecast(
    *,
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
    seq_len: int = 336,
    pred_len: int = 96,
    forecast_len: int = 512,
    num_forecasts: int = 10,
    n_block: int = 2,
    ff_dim: int = 2048,
    dropout: float = 0.05,
    norm_type: Literal["L", "B"] = "B",
    activation: Literal["relu", "gelu"] = "relu",
    kernel_size: int = 4,
    batch_size: int = 32,
    out_dir: str = ".",
    learning_rate: float = 0.0001,
    checkpoint_dir: str = "./checkpoints/",
    **_,
):
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
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

    model: tf.keras.Model
    # build model
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
    model.load_weights(os.path.join(out_dir, checkpoint_dir, f"{exp_id}_best"))
    starts = np.random.randint(
        len(data_loader.test_df) - forecast_len - seq_len, size=num_forecasts
    )

    all_inputs = np.concatenate(
        [
            next(
                iter(
                    data_loader._make_dataset(
                        data_loader.test_df.values[
                            start : start + seq_len + pred_len, :
                        ],
                        shuffle=False,
                    )
                )
            )[0]
            for start in starts
        ]
    )
    all_outputs = np.concatenate(
        [
            next(
                iter(
                    data_loader._make_dataset(
                        data_loader.test_df.values[
                            start : start + seq_len + pred_len, :
                        ],
                        shuffle=False,
                    )
                )
            )[1]
            for start in starts
        ]
    )
    inverse_transform = lambda df: np.array(
        [data_loader.inverse_transform(d) for d in df]
    )

    predictions = model(all_inputs)

    inputs = inverse_transform(all_inputs)
    labels = inverse_transform(all_outputs)
    predictions = inverse_transform(predictions)
    for i, start in enumerate(starts):
        for j in range(predictions.shape[-1]):
            actual = np.concatenate((inputs[i, :, j], predictions[i, :, j]))
            predicted = np.concatenate((inputs[i, :, j], labels[i, :, j]))
            plt.clf()
            df = pd.DataFrame.from_dict({"actual": actual, "predicted": predicted})
            df.columns = df.columns.set_names(data_loader.train_df.columns[j])
            sns.lineplot(df)
            plt.savefig(
                os.path.join(
                    out_dir,
                    "plots",
                    f"{start}_{data_loader.train_df.columns[j].replace('/','_')}.png",
                )
            )
