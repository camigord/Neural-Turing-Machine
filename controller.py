import tensorflow as tf
import numpy as np

class BaseController:

    def __init__(self, input_size, output_size, memory_read_heads, word_size, shift_range, batch_size=1):
        """
        constructs a controller as described in the "Neural Turing Machines" paper
        https://arxiv.org/abs/1410.5401

        Parameters:
        ----------
        input_size: int
            the size of the data input vector
        output_size: int
            the size of the data output vector
        memory_read_heads: int
            the number of read haeds in the associated external memory
        word_size: int
            the size of the word in the associated external memory
        shift_range: int
            allowed integer shifts
        batch_size: int
            the size of the input data batch [optional]
        """

        self.input_size = input_size
        self.output_size = output_size
        self.read_heads = memory_read_heads
        self.word_size = word_size
        self.batch_size = batch_size
        self.shift_range = shift_range

        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        has_recurrent_update = callable(getattr(self, 'update_state', None))
        has_get_state = callable(getattr(self, 'get_state', None))
        self.has_recurrent_nn =  has_recurrent_update and has_get_state

        # the actual size of the neural network input after flatenning and
        # concatenating the input vector with the previously read vectors from memory
        self.nn_input_size = self.word_size * self.read_heads + self.input_size

        # X Reading heads + 1 writing head: Key + strength + gate + gamma + shift | erase and add vectors
        self.interface_vector_size = (self.word_size + 3 + (2*self.shift_range + 1)) * (self.read_heads + 1) + 2*self.word_size

        # define network vars
        with tf.name_scope("controller"):
            self.network_vars()

            self.nn_output_size = None
            with tf.variable_scope("shape_inference"):
                self.nn_output_size = self.get_nn_output_size()

            self.initials()

    def initials(self):
        """
        sets the initial values of the controller transformation weights matrices
        this method can be overwritten to use a different initialization scheme
        """
        # defining internal weights of the controller
        self.interface_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
            name='interface_weights'
        )
        self.nn_output_weights = tf.Variable(
            tf.random_normal([self.nn_output_size, self.output_size], stddev=0.1),
            name='nn_output_weights'
        )
        self.mem_output_weights = tf.Variable(
            tf.random_normal([self.word_size * self.read_heads, self.output_size],  stddev=0.1),
            name='mem_output_weights'
        )

    def network_vars(self):
        """
        defines the variables needed by the internal neural network
        [the variables should be attributes of the class, i.e. self.*]
        """
        raise NotImplementedError("network_vars is not implemented")


    def network_op(self, X):
        """
        defines the controller's internal neural network operation

        Parameters:
        ----------
        X: Tensor (batch_size, word_size * read_haeds + input_size)
            the input data concatenated with the previously read vectors from memory

        Returns: Tensor (batch_size, nn_output_size)
        """
        raise NotImplementedError("network_op method is not implemented")


    def get_nn_output_size(self):
        """
        retrives the output size of the defined neural network

        Returns: int
            the output's size

        Raises: ValueError
        """

        input_vector =  np.zeros([self.batch_size, self.nn_input_size], dtype=np.float32)
        output_vector = None

        if self.has_recurrent_nn:
            output_vector,_ = self.network_op(input_vector, self.get_state())
        else:
            output_vector = self.network_op(input_vector)

        shape = output_vector.get_shape().as_list()

        if len(shape) > 2:
            raise ValueError("Expected the neural network to output a 1D vector, but got %dD" % (len(shape) - 1))
        else:
            return shape[1]


    def parse_interface_vector(self, interface_vector):
        """
        parses the flat interface_vector into its various components with their
        correct shapes

        Parameters:
        ----------
        interface_vector: Tensor (batch_size, interface_vector_size)
            the flattened interface vector to be parsed

        Returns: dict
            a dictionary with the components of the interface_vector parsed
        """

        parsed = {}

        r_keys_end = self.word_size * self.read_heads
        r_strengths_end = r_keys_end + self.read_heads
        r_gates_end = r_strengths_end + self.read_heads
        r_gamma_end = r_gates_end + self.read_heads
        r_shift_end = r_gamma_end + (self.shift_range * 2 + 1) * self.read_heads

        w_key_end = r_shift_end + self.word_size
        w_strengths_end = w_key_end + 1
        w_gates_end = w_strengths_end + 1
        w_gamma_end = w_gates_end + 1
        w_shift_end = w_gamma_end + (self.shift_range * 2 + 1)

        erase_end = w_shift_end + self.word_size
        write_end = erase_end + self.word_size

        r_keys_shape = (-1, self.word_size, self.read_heads)
        r_scalars_shape = (-1, self.read_heads)
        r_shift_shape = (-1, self.shift_range * 2 + 1, self.read_heads)

        w_key_shape = (-1, self.word_size, 1)
        w_scalars_shape = (-1, 1)
        w_shift_shape = (-1, self.shift_range * 2 + 1,1)

        write_shape = erase_shape = (-1, self.word_size)

        # parsing the vector into its individual components
        '''
        parsed['read_keys'] = tf.tanh(tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape))
        parsed['read_strengths'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_scalars_shape))
        parsed['read_gates'] = tf.sigmoid(tf.reshape(interface_vector[:, r_strengths_end:r_gates_end], r_scalars_shape))
        parsed['read_gammas'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_gates_end:r_gamma_end], r_scalars_shape)) + 1
        parsed['read_shifts'] = tf.nn.softmax(tf.reshape(interface_vector[:, r_gamma_end:r_shift_end], r_shift_shape),dim=1)

        parsed['write_key'] = tf.tanh(tf.reshape(interface_vector[:, r_shift_end:w_key_end], w_key_shape))
        parsed['write_strength'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_key_end:w_strengths_end], w_scalars_shape))
        parsed['write_gate'] = tf.sigmoid(tf.reshape(interface_vector[:, w_strengths_end:w_gates_end], w_scalars_shape))
        parsed['write_gamma'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_gates_end:w_gamma_end], w_scalars_shape)) + 1
        parsed['write_shift'] = tf.nn.softmax(tf.reshape(interface_vector[:, w_gamma_end:w_shift_end], w_shift_shape),dim=1)

        parsed['erase_vector'] = tf.sigmoid(tf.reshape(interface_vector[:, w_shift_end:erase_end], erase_shape))
        parsed['write_vector'] = tf.tanh(tf.reshape(interface_vector[:, erase_end:write_end], write_shape))
        '''
        parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape)
        parsed['read_strengths'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_scalars_shape))+1
        parsed['read_gates'] = tf.sigmoid(tf.reshape(interface_vector[:, r_strengths_end:r_gates_end], r_scalars_shape))
        parsed['read_gammas'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_gates_end:r_gamma_end], r_scalars_shape))+1
        parsed['read_shifts'] = tf.nn.softmax(tf.reshape(interface_vector[:, r_gamma_end:r_shift_end], r_shift_shape),dim=1)

        parsed['write_key'] = tf.reshape(interface_vector[:, r_shift_end:w_key_end], w_key_shape)
        parsed['write_strength'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_key_end:w_strengths_end], w_scalars_shape))+1
        parsed['write_gate'] = tf.sigmoid(tf.reshape(interface_vector[:, w_strengths_end:w_gates_end], w_scalars_shape))
        parsed['write_gamma'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_gates_end:w_gamma_end], w_scalars_shape)) + 1
        parsed['write_shift'] = tf.nn.softmax(tf.reshape(interface_vector[:, w_gamma_end:w_shift_end], w_shift_shape),dim=1)

        parsed['erase_vector'] = tf.sigmoid(tf.reshape(interface_vector[:, w_shift_end:erase_end], erase_shape))
        parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)

        return parsed

    def process_input(self, X, last_read_vectors, state=None):
        """
        processes input data through the controller network and returns the
        pre-output and interface_vector

        Parameters:
        ----------
        X: Tensor (batch_size, input_size)
            the input data batch
        last_read_vectors: (batch_size, word_size, read_heads)
            the last batch of read vectors from memory
        state: Tuple
            state vectors if the network is recurrent

        Returns: Tuple
            pre-output: Tensor (batch_size, output_size)
            parsed_interface_vector: dict
        """

        flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
        complete_input = tf.concat(1, [X, flat_read_vectors])
        nn_output, nn_state = None, None

        if self.has_recurrent_nn:
            nn_output, nn_state = self.network_op(complete_input, state)
        else:
            nn_output = self.network_op(complete_input)

        pre_output = tf.matmul(nn_output, self.nn_output_weights)
        interface = tf.matmul(nn_output, self.interface_weights)
        parsed_interface = self.parse_interface_vector(interface)

        if self.has_recurrent_nn:
            return pre_output, parsed_interface, nn_state
        else:
            return pre_output, parsed_interface


    def final_output(self, pre_output, new_read_vectors):
        """
        returns the final output by taking recent memory changes into account

        Parameters:
        ----------
        pre_output: Tensor (batch_size, output_size)
            the ouput vector from the input processing step
        new_read_vectors: Tensor (batch_size, words_size, read_heads)
            the newly read vectors from the updated memory

        Returns: Tensor (batch_size, output_size)
        """

        flat_read_vectors = tf.reshape(new_read_vectors, (-1, self.word_size * self.read_heads))

        final_output = pre_output + tf.matmul(flat_read_vectors, self.mem_output_weights)

        return final_output
