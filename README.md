# Neural-Turing-Machine

This is a TensorFlow implementation of [Neural Turing Machine](https://arxiv.org/abs/1410.5401). The code is inspired on the DNC implementation of [Mostafa-Samir](https://github.com/Mostafa-Samir/DNC-tensorflow) and it therefore follows the same structure and organization. However, some of the content addressing functions were adapted from the NTM implementation from [carpedm20](https://github.com/carpedm20/NTM-tensorflow).

The code is designed to deal with variable length inputs in a more efficient way than the reference code provided by [carpedm20](https://github.com/carpedm20/NTM-tensorflow).

The implementation currently supports only the _copy_ task described in the paper.

_[Important Notes]:_
1. For training efficiency, I replaced the circular convolution code proposed by [carpedm20](https://github.com/carpedm20/NTM-tensorflow) in order to use some already optimized TensorFlow functions. This code, however, was hard-coded for batch_sizes=1 and shift_range = 1. If you want to use other values, remember to either modify the proposed function (apply_conv_shift - memory.py) or to replace the function by the commented code.

2. In a similar way, the Hamming distance computation used in train.py for performance visualization is currently hard-coded for batch_sizes=1. Modifying this, however, should be much simpler...

3. I experienced the typical problem of the gradients becoming _nan_ when training with a max_sequence_length of 20. I have not been able to find any robust solution to this problem, but the code seems to converge when trained with a sequence length of 10. Some have suggested to perform _curriculum learning_ to avoid this issue, but I currently have not tried this option.

## Local Environment Specifications

The model was trained and tested on a machine with:
  - Intel® Core™ i7-2700K CPU @ 3.50GHz × 8
  - 16GB RAM
  - No GPU
  - Ubuntu 14.04 LTS
  - TensorFlow r0.12
  - Python 2.7

## Usage

To train a copy task:

`python train.py --iterations=100000`

## Results for the copy task

_You can generate similar results or play with the sequence length in the [visualization notebook](https://github.com/camigord/Neural-Turing-Machine/blob/master/Visualization.ipynb)._

It is really interesting to check the memory location map, where you can see that the network learns to address the memory in a sequential way and in the same order they were written into.

As I mentioned, this model was trained with a maximum sequence length of 10, but it can generalize to longer sequences (in this case sequence length = 30).

![results](https://github.com/camigord/Neural-Turing-Machine/blob/master/assets/results.png)

## Training loss

I trained the model with a recurrent controller, a maximum sequence length of 10 and an input size of 8 bits + 2 flags (start and finish) for 100000 iterations. The learning process can be seen below:

![hamming](https://github.com/camigord/Neural-Turing-Machine/blob/master/assets/Hamming.png)

![loss](https://github.com/camigord/Neural-Turing-Machine/blob/master/assets/Loss.png)
