# Neural-Turing-Machine

This is a TensorFlow implementation of [Neural Turing Machine](https://arxiv.org/abs/1410.5401). The code is inspired on the DNC implementation of [Mostafa-Samir](https://github.com/Mostafa-Samir/DNC-tensorflow) and it therefore tries to follow the same structure and organization. However, some of the content addressing functions were adapted from the NTM implementation from [carpedm20](https://github.com/carpedm20/NTM-tensorflow).

The code is designed to deal with variable length inputs in a way which I consider cleaner and more efficient than the reference code provided by [carpedm20](https://github.com/carpedm20/NTM-tensorflow).

The implementation currently supports only the _copy_ task described in the paper and employs a Feedforward controller (however, the code was designed to easily allow the use of recurrent controllers, see [Mostafa-Samir's repo](https://github.com/Mostafa-Samir/DNC-tensorflow/blob/master/docs/basic-usage.md)).

## Local Environment Specifications

The model was trained and tested on a machine with:
  - Intel® Core™ i7-2700K CPU @ 3.50GHz × 8
  - 16GB RAM
  - No GPU
  - Ubuntu 14.04 LTS
  - TensorFlow r0.12
  - Python 2.7

  
