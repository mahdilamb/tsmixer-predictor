# TSMixer forecasting

This package is a demonstration of how to train and use [TSMixer](https://github.com/google-research/google-research/tree/master/tsmixer) for forecasting.

## Installation

```shell
pip install -e .[all]
```

Alternatively, you don't need to include any optional dependencies (i.e. `pip install -e .`)

## Datasets

To download the datasets either:

```shell
make datasets
```

## Training

To train the models as per the paper run:

```shell
make train-weather [out_dir="."]
make train-ETTm2 [out_dir="."]
make train-electricity [out_dir="."]
make train-traffic [out_dir="."]
```

To train the model with custom parameters, refer to the help via:

```shell
python3 -m tsmixer train --help
```

## Example

For the following example, we train on the weather dataset. Our losses and errors are as follow:

<!--{$ plot= tsmixer.utils.plot:plot_tensorboard_scalar("weather/logs", "epoch_loss")$}--><!--{{plot}}-->
![Training and validation loss](./assets/tensorboard_epoch_loss.png "Training and validation loss")<!--{><}-->
<!--{$ plot= tsmixer.utils.plot:plot_tensorboard_scalar("weather/logs", "epoch_mae")$}--><!--{{plot}}-->
![Training and validation loss](./assets/tensorboard_epoch_mae.png "Training and validation loss")<!--{><}-->

### Sample forecasts

Forecasts were run using:

```shell
make forecast out_dir="./weather"
```
<!--{$ plots = glob:glob("./weather/plots/*.png") $}-->
<!--{% for path in plots %}-->
<!--{$  plot= tsmixer.utils.plot:copy_and_markdown(path) $}-->
<!--{{ plot}}-->
<!--{% endfor %}-->
![](assets/plots/2732_Tdew%20(degC).png)

![](assets/plots/2732_PAR%20(�mol_m�_s).png)

![](assets/plots/2732_wd%20(deg).png)

![](assets/plots/2732_wv%20(m_s).png)

![](assets/plots/2732_rain%20(mm).png)

![](assets/plots/2732_max.%20wv%20(m_s).png)

![](assets/plots/2732_rh%20(%).png)

![](assets/plots/2732_H2OC%20(mmol_mol).png)

![](assets/plots/2732_VPact%20(mbar).png)

![](assets/plots/2732_p%20(mbar).png)

![](assets/plots/2732_SWDR%20(W_m�).png)

![](assets/plots/2732_raining%20(s).png)

![](assets/plots/2732_sh%20(g_kg).png)

![](assets/plots/2732_VPdef%20(mbar).png)

![](assets/plots/2732_VPmax%20(mbar).png)

![](assets/plots/2732_Tlog%20(degC).png)

![](assets/plots/2732_rho%20(g_m**3).png)

![](assets/plots/2732_Tpot%20(K).png)

![](assets/plots/2732_T%20(degC).png)

![](assets/plots/2732_max.%20PAR%20(�mol_m�_s).png)

![](assets/plots/2732_OT.png)
<!--{><}-->
