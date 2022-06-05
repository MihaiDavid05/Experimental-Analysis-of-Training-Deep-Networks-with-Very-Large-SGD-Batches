# EPFL OML Project

### Abstract
Bla bla bla

### Environment setup
For our experiments we used Google Colab.
Tested configurations: 
* `Python 3.7`
* `pytorch ???`
* `torchvision ???`
* `CUDA` according to Colab 
* `cuDNN` according to Colab 

### 1.Libraries installation
To install PyTorch (CPU only) use the following command:
```bash
pip install pytorch torchvision torchaudio cpuonly -c pytorch
```
For PyTorch and torchvision with CUDA installation use the following command:
```bash
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
```

Other necessary installations:
```bash
pip install numpy tensorboard tqdm future pillow
```

### 2.Set PYTHONPATH
Please add the project root folder to `$PYTHONPATH` using following command:
```bash
export PYTHONPATH=$PYTHONPATH:<path_to_project_folder>
```

### 3.Data and folder structure
Please download python version of CIFAR10 and CIFAR100 at this [link](https://www.cs.toronto.edu/~kriz/cifar.html).
Unzip the contents under the `data` folder.
```
└── PROJECT_ROOT
       ├── data                 <- dataset
       |   ├── cifar-10-batches-py
       |   └── cifar-100-python
       ├── checkpoints          <- models weights    
       ├── configs              <- configuration files
       ├── logs                 <- experiments log files
       ├── src                  <- train, predict and data augmentation scripts
       └── utils                <- multiple utility scripts grouped by functionality
```

### 4.Tensorboard
You may encouter an error with tensorboard. Install this:
```bash
pip install setuptools==59.5.0
```

To run Tensorboard, run this:
```bash
python -m tensorboard.main --logdir=logs/<log_folder_wanted>
```

### 5.Training and validation
Bla bla

### 6.Prediction
Bla bla 

### 7. Results
Bla bla



