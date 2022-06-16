# EPFL OML Project

### Abstract
TO BE DONE!

### Environment setup
For our experiments we used Google Colab. We provide the notebook under `OML_notebook.ipynb`.
It has 4 parts: Setup, Training, Event Parsing and Testing. You can find more guidance in the notebook.

However, we also made our own local environment for quick experiments and better debugging (CPU only).
Tested configurations locally: 
* `Python 3.7`
* `pytorch 1.9.0` (CPU only)
* `torchvision 0.10.0`

### 1.Libraries installation
To install PyTorch (CPU only) use the following command:
```bash
pip install pytorch torchvision torchaudio cpuonly -c pytorch
```

Other necessary installations:
```bash
pip install numpy tensorboard tqdm pillow pandas
pip install git+https://github.com/lehduong/torch-warmup-lr.git
```

You may encouter an error with tensorboard at runtime, when working locally. Install this:
```bash
pip install setuptools==59.5.0
pip install tensorflow==2.9.1
```

### 2.Set PYTHONPATH
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```

### 3.Data and folder structure
Please download the python version of CIFAR10 at this [link](https://www.cs.toronto.edu/~kriz/cifar.html).
Unzip the contents under the `data` folder.
```
└── PROJECT_ROOT
       ├── data                 <- dataset
       |   ├── cifar-10-batches-py
       |   └── cifar-100-python
       ├── checkpoints          <- models weights    
       ├── configs              <- configuration files
       |   ├── experiment1
           | ....
       |   └── experiment6
       ├── logs                 <- experiments log files
       ├── src                  <- train, predict and event parser scripts
       ├── utils                <- multiple utility scripts
       ├── experiments.py       <- script used in bash scripts, for training
       └── run_expX.sh          <- bash for deploying a specific suite of experiments on a remote GPU
```

### 4.Tensorboard
To run Tensorboard, run this:
```bash
python -m tensorboard.main --logdir=logs/<log_folder_wanted>
```

### 5.Training and validation
For training and validation you can run this command:
```shell script
cd src
python run.py <config_filename>
```
Example:
```shell script
cd src
python run.py "config2"
```
NOTE: If training locally, you should define a `configX.json` file right under `configs` folder.

### 6.Prediction
For prediction on test set you can run this command:
```shell script
cd src
python test.py <config_filename>
```
Example:
```shell script
cd src
python test.py "config2"
```
NOTE: If testing locally, you should define a `configX.json` file right under `configs` folder.

### 7. Results
TO BE DONE!



