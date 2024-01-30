# RandomAugmentation

## Setup
Install the conda environment from yml file. Python 3.8 is used in this project.

```
conda env create -f environment.yml
```

Activate the conda environment running the following command.

```
conda activate detectron
```

To install detectron in your conda environment

```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Run
To run the code first activate the conda environment.

```
conda activate detectron
```

To start the training run the following command. Amount of GPUs needs to be adjusted depending on your system.

```
python train_net.py --num-gpus 2
```

To modify the training the config file needs to be edited. For more information read the detectron2 docs: https://detectron2.readthedocs.io/en/latest/index.html
