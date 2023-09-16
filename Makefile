.PHONY: help tests black isort docstrings format mypy qc refresh-models download-models datasets train-weather train-electricity train-ETTm2 train-traffic zip-file forecast
default: help

MAIN_PACKAGE_DIRECTORY="tsmixer"
PYTHON_SRC_DIRECTORIES=${MAIN_PACKAGE_DIRECTORY} "tests"

tests: # Run tests using pytest.
	pytest --cov=${MAIN_PACKAGE_DIRECTORY} --cov-report term-missing

black: # Format the python files using black.
	black ${PYTHON_SRC_DIRECTORIES}

isort: # Sort the imports in python files.
	isort ${PYTHON_SRC_DIRECTORIES}

bandit: # Run security checks with bandit.
	bandit -r ${MAIN_PACKAGE_DIRECTORY}

docstrings:  # Format the docstrings using docformatter
	docformatter --in-place -r ${PYTHON_SRC_DIRECTORIES};  pydocstyle ${PYTHON_SRC_DIRECTORIES}

format: isort black docstrings # Format the source files with isort and black.

mypy: # Check types with mypy.
	mypy ${MAIN_PACKAGE_DIRECTORY}

qc: format tests bandit mypy # Run all the QC tasks.

refresh-models: # Copy the models from github
	command -v svn || sudo $(command -v yum || command -v apt-get) install subversion;\
	svn export https://github.com/google-research/google-research/trunk/tsmixer/tsmixer_basic/models tsmixer/models/tsmixer_basic;\
	python -c 'import tsmixer.models.tsmixer_basic.tsmixer_rev_in' || echo "Module paths are incorrect, please re-check code."

download-models: # Download the models from github
	command -v svn || sudo $(command -v yum || command -v apt-get) install subversion;\
	svn export https://github.com/google-research/google-research/trunk/tsmixer/tsmixer_basic downloads/tsmixer_basic --force

datasets: # Download the datasets
	sh ./scripts/download_datasets.sh

train-weather: out_dir="."
train-weather: 
	@[ -d dataset ] || ./scripts/download_datasets.sh; \
	python3 -m tsmixer train --model tsmixer_rev_in --data weather --out_dir ${out_dir} --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 4 --dropout 0.3 --ff_dim 32

train-ETTm2: out_dir="."
train-ETTm2: 
	@[ -d dataset ] || ./scripts/download_datasets.sh; \
	python3 -m tsmixer train --model tsmixer_rev_in --data ETTm2 --out_dir ${out_dir} --seq_len 512 --pred_len 96 --learning_rate 0.001 --n_block 2 --dropout 0.9 --ff_dim 64


train-electricity: out_dir="."
train-electricity: 
	@[ -d dataset ] || ./scripts/download_datasets.sh; \
	python3 -m tsmixer train --model tsmixer_rev_in --data electricity --out_dir ${out_dir} --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 4 --dropout 0.7 --ff_dim 64


train-traffic: out_dir="."
train-traffic: 
	@[ -d dataset ] || ./scripts/download_datasets.sh; \
	python3 -m tsmixer train --model tsmixer_rev_in --data traffic --out_dir ${out_dir} --seq_len 512 --pred_len 96 --learning_rate 0.0001 --n_block 8 --dropout 0.7 --ff_dim 64

forecast: out_dir="." num_forecasts=10
forecast: 
	python3 -m tsmixer forecast --out_dir ${out_dir} --num_forecasts ${num_forecasts}


zip-file:
	zip ${file} $(shell git ls-files)

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m\n\t$$(echo $$l | cut -f 2- -d'#')\n"; done
