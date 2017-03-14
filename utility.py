import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_state_ops

def unpack_into_tensorarray(value, axis, size=None):
    """
    unpacks a given tensor along a given axis into a TensorArray

    Parameters:
    ----------
    value: Tensor
        the tensor to be unpacked
    axis: int
        the axis to unpack the tensor along
    size: int
        the size of the array to be used if shape inference resulted in None

    Returns: TensorArray
        the unpacked TensorArray
    """

    shape = value.get_shape().as_list()
    rank = len(shape)
    dtype = value.dtype
    array_size = shape[axis] if not shape[axis] is None else size

    if array_size is None:
        raise ValueError("Can't create TensorArray with size None")

    array = tf.TensorArray(dtype=dtype, size=array_size)
    dim_permutation = [axis] + range(1, axis) + [0] + range(axis + 1, rank)
    unpack_axis_major_value = tf.transpose(value, dim_permutation)
    full_array = array.unpack(unpack_axis_major_value)

    return full_array

def pack_into_tensor(array, axis):
    """
    packs a given TensorArray into a tensor along a given axis

    Parameters:
    ----------
    array: TensorArray
        the tensor array to pack
    axis: int
        the axis to pack the array along

    Returns: Tensor
        the packed tensor
    """

    packed_tensor = array.pack()
    shape = packed_tensor.get_shape()
    rank = len(shape)

    dim_permutation = [axis] + range(1, axis) + [0]  + range(axis + 1, rank)
    correct_shape_tensor = tf.transpose(packed_tensor, dim_permutation)

    return correct_shape_tensor
