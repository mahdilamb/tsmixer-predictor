import glob
import os
import shutil
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.summary import summary_iterator


def plot_tensorboard_scalar(path: str, scalar_name: str):
    """Plot a scalar from tensor flow logs."""
    output = f"./assets/tensorboard_{scalar_name}.png"
    plt.clf()
    logs = glob.glob(os.path.join(path, "*", "events.out.tfevents.*"))
    values = {}
    for log in logs:
        _, which, _ = log.rsplit(os.path.sep, maxsplit=2)
        data: list[float] = []
        for e in summary_iterator.summary_iterator(log):
            for v in e.summary.value:
                if v.tag == scalar_name:
                    data.append(float(tf.make_ndarray(v.tensor)))
        values[which] = data
    ax: plt.Axes = sns.lineplot(pd.DataFrame.from_dict(values))
    ax.set_ylabel(scalar_name.replace("_", " ").title())
    ax.set_xlabel("Loss")
    plt.savefig(output)
    return f'\n![Training and validation loss]({output.replace(" ", "%20")} "Training and validation loss")'


def plot_features(data: str, features: Sequence[str] | None = None):
    """Plot the features from a path.

    If no features provided, plot all.
    """
    output = f"./assets/features_{data}.png"
    df = pd.read_csv(os.path.join("dataset", f"{data}.csv"))
    if features is None:
        features = df.columns[1:].to_list()
    plot_df = df[features]
    plot_df.index = pd.to_datetime(df.iloc[:, 0])
    plt.clf()
    plot_df.plot(subplots=True)
    plt.savefig(output)
    return f'\n![Time series data of {data}]({output.replace(" ", "%20")} "Time series data of {data}")\n'


def describe_data(data: str):
    """Describe the data as markdown."""
    return (
        "\n"
        + pd.read_csv(os.path.join("dataset", f"{data}.csv"))
        .describe()
        .transpose()
        .to_markdown()
        + "\n"
    )


def plot_frequency(data: str, feature: str):
    """Plot frequency features."""
    output = f"./assets/frequency_{data}_{feature}.png"
    df = pd.read_csv(os.path.join("dataset", f"{data}.csv"))
    plt.clf()
    fft = tf.signal.rfft(df[feature])
    f_per_dataset = np.arange(0, len(fft))
    sample_frequency = pd.to_datetime(df.iloc[:, 0]).diff().value_counts()
    sample_frequency = 3600 / sample_frequency.index[sample_frequency.argmax()].seconds
    n_samples_h = len(df[feature]) / sample_frequency
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / (hours_per_year)

    f_per_year = f_per_dataset / years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale("log")
    plt.xticks([1, 365.2524], labels=["1/Year", "1/day"])
    _ = plt.xlabel("Frequency (log scale)")
    plt.savefig(output)
    return f'\n![Time frequency feature of {feature} from {data}.]({output.replace(" ", "%20")})\n'


def copy_and_markdown(path: str):
    """Copy a file to assets and print markdown."""
    os.makedirs(os.path.join("assets", "plots"), exist_ok=True)
    output = os.path.join("assets", "plots", os.path.basename(path))

    shutil.copyfile(path, output)

    return f"""
![]({output.replace(" ", "%20")})
"""
