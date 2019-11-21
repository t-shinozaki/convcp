# convcp

## test environment
- CentOS 7.6
- Python 2.7


## Requirements
- Chainer v.4.x
- Cupy v.4.x
- datasets
  - MNIST
  - CIFAR
  - ImageNet


# setup virtualenv
    cd
    mkdir envs
    virtualenv --system-site-packages envs/chainer
    source envs/chainer/bin/activate
    pip install -U setuptools pip
    pip install 'cupy<5.0' 'chainer<5.0' 
    pip install Pillow matplotlib tqdm h5py Cython


# Download MNIST & CIFAR-10 datasets
    python prep_data.py


## MNIST
### baseline by BP learning
    python train_cifar.py -g 0 -i 3000 -c 10 -n \
        -D mnist.npz -M mnist -K fc.conv1

### competitive learning
    python train_cifar_cp.py -g 0 -i 600 -E 20 -C 1 \
        -D mnist.npz -M mnist_cp -K model.conv2 -O mnist_cp
    python train_cifar.py -g 0 -i 3000 -c 10 -n \
        -D mnist.npz -M mnist_cp -W ../mnist_cp/model_fin.h5


## CIFAR
### baseline by BP learning
    python train_cifar.py -g 0 -i 15000 -n \
        -M lenet -K fc.conv1

### competitive learning
    python train_cifar_cp.py -g 0 -E 200 \
        -M lenet_cp -K model.conv2 -O lenet_cp
    python train_cifar.py -g 0 -i 15000 -n \
        -M lenet_cp -W ../lenet_cp/model_fin.h5


## ImageNet
### competitive learning
    python train_alex_cp.py -g 0 -E 300 -C 10 -s 100 -B 8 -M alex_cp -O alex_cp
	
### fine tuning
    python train_alex.py -g 0 -i 20000 -n \
        -M alex_cp -W ../alex_cp/model_fin_cp.h5 -O alex_ft1 \
        --lr 0.01 
    python train_alex.py -g 0 -i 20000 -n \
        -M alex_cp -W ../alex_cp/model_fin_cp.h5 -O alex_ft2 \
        --lr 0.001 -F alex_ft1/model_fin_bp_ft.h5 
    python train_alex.py -g 0 -i 20000 -n \
        -M alex_cp -W ../alex_cp/model_fin_cp.h5 -O alex_ft3 \
        --lr 0.0001 -F alex_ft2/model_fin_bp_ft.h5 

### test top-5 error
    python test_alex_top5.py -g 0 \
	-M alex_cp -W ../alex_cp/model_fin_cp.h5 \
        -F alex_ft3/model_fin_bp_ft.h5
        
