# EPFL OML Project

### Abstract
TO BE DONE!

### Environment setup
For our experiments we used Google Colab. We provide the notebook under `OML_notebook.ipynb`.

However, we also made our own local environment (CPU only).
Tested configurations locally: 
* `Python 3.7`
* `pytorch 1.9.0` (CPU only)
* `torchvision 0.10.0`

### 1.Libraries installation
To install PyTorch (CPU only) use the following command:
```bash
pip install pytorch torchvision torchaudio cpuonly -c pytorch
```
For PyTorch and torchvision with CUDA installation use the following command (NOT TESTED):
```bash
conda install pytorch torchvision cudatoolkit=<compatible_version> -c pytorch
```
NOTE: Search for a compatible cudatoolkit version with your local GPU.

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
Please download the python version of CIFAR10 and CIFAR100 at this [link](https://www.cs.toronto.edu/~kriz/cifar.html).
Unzip the contents under the `data` folder.
```
└── PROJECT_ROOT
       ├── data                 <- dataset
       |   ├── cifar-10-batches-py
       |   └── cifar-100-python
       ├── checkpoints          <- models weights    
       ├── configs              <- configuration files
       ├── logs                 <- experiments log files
       ├── src                  <- train, predict and event parser scripts
       └── utils                <- multiple utility scripts
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

### 7. Results
TO BE DONE!



