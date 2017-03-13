import tensorflow as tf
import numpy as np
import math

class Memory:

    def __init__(self, memory_locations=128, word_size=20, read_heads=1, batch_size=1):
        """
        constructs a memory matrix with read heads and a write head as described
        in the "Neural Turing Machines" paper
        https://arxiv.org/abs/1410.5401

        Parameters:
        ----------
        memory_locations: int
            the number of memory locations
        word_size: int
            the vector size at each location
        read_heads: int
            the number of read heads that can read simultaneously from the memory
        batch_size: int
            the size of input data batch
        """

        self.memory_locations = memory_locations
        self.word_size = word_size
        self.read_heads = read_heads
        self.batch_size = batch_size

    def init_memory(self):
        """
        returns the initial values for the memory Parameters

        Returns: Tuple
        """

        return (
            tf.fill([self.batch_size, self.memory_locations, self.word_size], 1e-6),  # initial memory matrix
            tf.fill([self.batch_size, self.memory_locations, ], 1e-6),  # initial write weighting
            tf.fill([self.batch_size, self.memory_locations, self.read_heads], 1e-6),  # initial read weightings
            tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6),  # initial read vectors
        )

    def get_content_adressing(self, memory_matrix, keys, strengths):
        """
        retrives a content-based addressing weighting given the keys

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, memory_locations, word_size)
            the memory matrix to lookup in
        keys: Tensor (batch_size, word_size, number_of_keys)
            the keys to query the memory with
        strengths: Tensor (batch_size, number_of_keys, )
            the list of strengths for each lookup key

        Returns: Tensor (batch_size, memory_locations, number_of_keys)
            The list of lookup weightings for each provided key
        """

        normalized_memory = tf.nn.l2_normalize(memory_matrix, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)

        similiarity = tf.batch_matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)

        return tf.nn.softmax(similiarity * strengths, 1)

    def apply_interpolation(self,content_weights,prev_weights,interpolation_gate):
        """
        retrives a location-based addressing given the interpolation_gate

        Parameters:
        ----------
        content_weights: Tensor (batch_size, memory_locations, number_of_keys)
            content_based addressing weights
        prev_weights: Tensor (batch_size, memory_locations)
            the write_weighting from the last time step
        interpolation_gate: Tensor (batch_size, 1)
            blending factor g

        Returns: Tensor (batch_size, memory_locations, number_of_keys)
            The gated weighting
        """

        gated_weighting = interpolation_gate * content_weights + (1 - interpolation_gate) * prev_weights

        return gated_weighting

    def apply_conv_shift(self,gated_weighting,shift_weighting):
        """
        applying rotation over the gated weights

        Parameters:
        ----------
        gated_weighting: Tensor (batch_size, memory_locations, number_of_keys)
            The gated weighting
        shift_weighting: Tensor (batch_size, ?)
            distribution over the allowed integer shifts

        Returns: Tensor (batch_size, memory_locations, number_of_keys)
            weights after circular Convolution
        """

        size = int(gated_weighting.get_shape()[1])
        kernel_size = int(shift_weighting.get_shape()[0])
        kernel_shift = int(math.floor(kernel_size/2.0))

        def loop(idx):
            if idx < 0: return size + idx
            if idx >= size : return idx - size
            else: return idx

        kernels = []
        for i in xrange(size):
            indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
            v_ = tf.transpose(tf.gather(tf.transpose(gated_weighting), indices))
            kernels.append(tf.reduce_sum(v_ * shift_weighting, 1))

        return tf.transpose(tf.stack(kernels, axis=0))

    def sharp_weights(self,after_conv_shift, sharp_gamma):
        """
        Sharpens the final weights

        Parameters:
        ----------
        after_conv_shift: Tensor (batch_size, memory_locations, number_of_keys)
            weights after circular Convolution
        sharp_gamma: Tensor (batch_size, 1)
            scalar to sharpen the final weights

        Returns: Tensor (batch_size, memory_locations, number_of_keys)
            final weights
        """

        powed_conv_w = tf.pow(after_conv_shift, sharp_gamma)
        return powed_conv_w / tf.expand_dims(tf.reduce_sum(powed_conv_w,1),1)

    def update_memory(self, memory_matrix, write_weighting, add_vector, erase_vector):
        """
        updates and returns the memory matrix given the weighting, add and erase vectors
        and the memory matrix from previous step

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, memory_locations, word_size)
            the memory matrix from previous step
        write_weighting: Tensor (batch_size, memory_locations)
            the weight of writing at each memory location
        add_vector: Tensor (batch_size, word_size)
            a vector specifying what to write
        erase_vector: Tensor (batch_size, word_size)
            a vector specifying what to erase from memory

        Returns: Tensor (batch_size, memory_locations, word_size)
            the updated memory matrix
        """

        # expand data with a dimension of 1 at multiplication-adjacent location
        # to force matmul to behave as an outer product
        write_weighting = tf.expand_dims(write_weighting, 2)
        add_vector = tf.expand_dims(add_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)

        erasing = memory_matrix * (1 - tf.batch_matmul(write_weighting, erase_vector))
        writing = tf.batch_matmul(write_weighting, add_vector)
        updated_memory = erasing + writing

        return updated_memory

    def update_read_vectors(self, memory_matrix, read_weightings):
        """
        reads, updates, and returns the read vectors of the recently updated memory

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, memory_locations, word_size)
            the recently updated memory matrix
        read_weightings: Tensor (batch_size, memory_locations, read_heads)
            the amount of info to read from each memory location by each read head

        Returns: Tensor (word_size, read_heads)
        """

        updated_read_vectors = tf.batch_matmul(memory_matrix, read_weightings, adj_x=True)

        return updated_read_vectors

    def write(self, memory_matrix, write_weighting, key, strength, interpolation_gate, shift_weighting,
              sharp_gamma, add_vector, erase_vector):
        """
        defines the complete pipeline of writing to memory given the write variables,
        the memory_matrix and the corresponding vectors from the previous step

        Parameters:
        ----------
        memory_matrix: Tensor (batch_size, memory_locations, word_size)
            the memory matrix from previous step
        write_weighting: Tensor (batch_size, memory_locations)
            the write_weighting from the last time step
        key: Tensor (batch_size, word_size, 1)
            the key to query the memory location with
        strength: Tensor (batch_size, 1)
            the strength of the query key (beta)
        interpolation_gate: Tensor (batch_size, 1)
            blending factor g
        shift_weighting: Tensor (batch_size, ?)
            distribution over the allowed integer shifts
        sharp_gamma: Tensor (batch_size, 1)
            scalar to sharpen the final weights
        add_vector: Tensor (batch_size, word_size)
            specifications of what to add to memory
        erase_vector: Tensor(batch_size, word_size)
            specifications of what to erase from memory

        Returns : Tuple
            the updated write_weighting: Tensor(batch_size, memory_locations)
            the updated memory_matrix: Tensor (batch_size, memory_locations, word_size)
        """

        content_addressed_w = self.get_content_adressing(memory_matrix, key, strength)
        content_addressed_w = tf.squeeze(content_addressed_w,axis=2)
        gated_weighting = self.apply_interpolation(content_addressed_w,write_weighting,interpolation_gate)
        after_conv_shift = self.apply_conv_shift(gated_weighting,shift_weighting)
        new_write_weighting = self.sharp_weights(after_conv_shift, sharp_gamma)

        # Write in memory
        new_memory_matrix = self.update_memory(memory_matrix, new_write_weighting, add_vector, erase_vector)

        return new_write_weighting, new_memory_matrix


    def read(self, memory_matrix, read_weightings, key, strength, interpolation_gate, shift_weighting,
             sharp_gamma):
        """
        defines the complete pipeline for reading from memory

        Parameters:
        ---------
        memory_matrix: Tensor (batch_size, memory_locations, word_size)
            the memory matrix from previous step
        read_weightings: Tensor (batch_size, memory_locations)
            the read_weightings from the last time step
        key: Tensor (batch_size, word_size, 1)
            the key to query the memory location with
        strength: Tensor (batch_size, 1)
            the strength of the query key (beta)
        interpolation_gate: Tensor (batch_size, 1)
            blending factor g
        shift_weighting: Tensor (batch_size, ?)
            distribution over the allowed integer shifts
        sharp_gamma: Tensor (batch_size, 1)
            scalar to sharpen the final weights

        Returns: Tuple
            the updated read_weightings: Tensor(batch_size, memory_locations, read_heads)
            the recently read vectors: Tensor (batch_size, word_size, read_heads)
        """

        content_addressed_w = self.get_content_adressing(memory_matrix, key, strength)
        gated_weighting = self.apply_interpolation(content_addressed_w,read_weightings,interpolation_gate)
        after_conv_shift = self.apply_conv_shift(gated_weighting,shift_weighting)
        new_read_weightings = self.sharp_weights(after_conv_shift, sharp_gamma)

        new_read_vectors = self.update_read_vectors(memory_matrix, new_read_weightings)

        return new_read_weightings, new_read_vectors
