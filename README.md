[![Build Status](https://travis-ci.com/EmanuelOverflow/ELM-TF.svg?branch=master)](https://travis-ci.com/EmanuelOverflow/ELM-TF)
---

# Extreme Learning Machine 

A simple implementation of **ELM** written in **Tensorflow**.

### Usage

To test the code download MNIST dataset and give it the path to the directory:

```
python train_mnist.py --dataset_path MNIST_data
```

You can change the batch size (default is 5000):

```
python train_mnist.py --dataset_path MNIST_data --batch_size 5000
```

It is possible to choose how many hidden units to use from command line (default is 512):

```
python train_mnist.py --dataset_path MNIST_data --num_hidden 512
```

Default settings give:

##### Training

``` 
Time: 0.8274s (on CPU)
Loss: 1.689
Acc: 0.945
```

##### Testing

```
Loss: 1.714
Acc: 0.903
```

### Reference

Huang, Guang-Bin, Qin-Yu Zhu, and Chee-Kheong Siew. "Extreme learning machine: theory and applications." Neurocomputing 70.1 (2006): 489-501.

http://www.sciencedirect.com/science/article/pii/S0925231206000385