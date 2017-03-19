import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memory import Memory
import utility
import os

class NTM:

    def __init__(self, controller_class, input_size, output_size, memory_locations = 256,
                 memory_word_size = 64, memory_read_heads = 4, shift_range=1, batch_size = 1):
        """
        constructs a complete DNC architecture as described in the "Neural Turing Machines" paper
        https://arxiv.org/abs/1410.5401

        Parameters:
        -----------
        controller_class: BaseController
            a concrete implementation of the BaseController class
        input_size: int
            the size of the input vector
        output_size: int
            the size of the output vector
        memory_locations: int
            the number of words that can be stored in memory
        memory_word_size: int
            the size of an individual word in memory
        memory_read_heads: int
            the number of read heads in the memory
        shift_range: int
            allowed integer shifts
        batch_size: int
            the size of the data batch
        """

        self.input_size = input_size
        self.output_size = output_size
        self.memory_locations = memory_locations
        self.word_size = memory_word_size
        self.read_heads = memory_read_heads
        self.batch_size = batch_size
        self.shift_range = shift_range

        self.memory = Memory(self.memory_locations, self.word_size, self.read_heads, self.batch_size)
        self.controller = controller_class(self.input_size, self.output_size, self.read_heads, self.word_size, self.shift_range, self.batch_size)

        # input data placeholders
        with tf.name_scope("Input"):
            self.input_data = tf.placeholder(tf.float32, [batch_size, None, input_size], name='input')
            self.target_output = tf.placeholder(tf.float32, [batch_size, None, output_size], name='targets')
            self.sequence_length = tf.placeholder(tf.int32, name='sequence_length')
            self.unpacked_input_data = utility.unpack_into_tensorarray(self.input_data, 1, self.sequence_length)

        self.build_graph()

    def build_graph(self):
        """
        builds the computational graph that performs a step-by-step evaluation
        of the input data batches
        """
        with tf.name_scope("Outputs") as outputs_scope:
            outputs = tf.TensorArray(tf.float32, self.sequence_length)
            read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
            write_weightings = tf.TensorArray(tf.float32, self.sequence_length)
            write_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        with tf.name_scope("Controller_State") as controller_scope:
            controller_state = self.controller.get_state() if self.controller.has_recurrent_nn else (tf.zeros(1), tf.zeros(1))
            if not isinstance(controller_state, LSTMStateTuple):
                controller_state = LSTMStateTuple(controller_state[0], controller_state[1])

        memory_state = self.memory.init_memory()

        final_results = None

        with tf.variable_scope("Sequence_Loop") as scope:
            time = tf.constant(0, dtype=tf.int32)

            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body,
                loop_vars=(
                    time, memory_state, outputs,
                    read_weightings, write_weightings, controller_state, write_vectors
                ),
                parallel_iterations=32,
                swap_memory=True
            )

        with tf.name_scope(controller_scope):
            dependencies = []
            if self.controller.has_recurrent_nn:
                dependencies.append(self.controller.update_state(final_results[5]))

        with tf.name_scope(outputs_scope):
            with tf.name_scope("Tensor_Arrays"):
                with tf.control_dependencies(dependencies):
                    self.packed_output = utility.pack_into_tensor(final_results[2], axis=1)
                    self.packed_memory_view = {
                        'read_weightings': utility.pack_into_tensor(final_results[3], axis=1),
                        'write_weightings': utility.pack_into_tensor(final_results[4], axis=1),
                        'write_vectors': utility.pack_into_tensor(final_results[6], axis=1)
                    }

    def _loop_body(self, time, memory_state, outputs, read_weightings, write_weightings, controller_state, write_vectors):
        """
        the body of the DNC sequence processing loop

        Parameters:
        ----------
        time: Tensor
        memory_state: Tuple
        outputs: TensorArray
        read_weightings: TensorArray,
        write_weightings: TensorArray,
        controller_state: Tuple

        Returns: Tuple containing all updated arguments
        """

        step_input = self.unpacked_input_data.read(time)
        output_list = self._step_op(step_input, memory_state, controller_state)

        # update memory parameters
        new_controller_state = tf.zeros(1)
        new_memory_state = tuple(output_list[0:4])

        new_controller_state = LSTMStateTuple(output_list[5], output_list[6])

        outputs = outputs.write(time, output_list[4])

        # collecting memory view for the current step
        read_weightings = read_weightings.write(time, output_list[2])
        write_weightings = write_weightings.write(time, output_list[1])
        write_vectors = write_vectors.write(time, output_list[7])

        return (
            time + 1, new_memory_state, outputs,
            read_weightings, write_weightings,
            new_controller_state, write_vectors
        )


    def _step_op(self, step, memory_state, controller_state=None):
        """
        performs a step operation on the input step data

        Parameters:
        ----------
        step: Tensor (batch_size, input_size)
        memory_state: Tuple
            a tuple of current memory parameters
        controller_state: Tuple
            the state of the controller if it's recurrent

        Returns: Tuple
            output: Tensor (batch_size, output_size)
            memory_view: dict
        """

        last_read_vectors = memory_state[3]
        pre_output, interface, nn_state = None, None, None

        if self.controller.has_recurrent_nn:
            pre_output, interface, nn_state = self.controller.process_input(step, last_read_vectors, controller_state)
        else:
            pre_output, interface = self.controller.process_input(step, last_read_vectors)

        write_weighting, memory_matrix = self.memory.write(
            memory_state[0], memory_state[1],
            interface['write_key'],
            interface['write_strength'],
            interface['write_gate'],
            interface['write_shift'],
            interface['write_gamma'],
            interface['write_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,
            memory_state[2],
            interface['read_keys'],
            interface['read_strengths'],
            interface['read_gates'],
            interface['read_shifts'],
            interface['read_gammas'],
        )

        return [
            # report new memory state to be updated outside the condition branch
            memory_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            pre_output,
            #self.controller.final_output(pre_output, read_vectors),

            # report new state of RNN if exists
            nn_state[0] if nn_state is not None else tf.zeros(1),
            nn_state[1] if nn_state is not None else tf.zeros(1),
            interface['write_vector']
        ]

    def get_outputs(self):
        """
        returns the graph nodes for the output and memory view

        Returns: Tuple
            outputs: Tensor (batch_size, time_steps, output_size)
            memory_view: dict
        """
        return self.packed_output, self.packed_memory_view


    def save(self, session, ckpts_dir, name):
        """
        saves the current values of the model's parameters to a checkpoint

        Parameters:
        ----------
        session: tf.Session
            the tensorflow session to save
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        checkpoint_dir = os.path.join(ckpts_dir, name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        tf.train.Saver(tf.trainable_variables()).save(session, os.path.join(checkpoint_dir, 'model.ckpt'))


    def restore(self, session, ckpts_dir, name):
        """
        session: tf.Session
            the tensorflow session to restore into
        ckpts_dir: string
            the path to the checkpoints directories
        name: string
            the name of the checkpoint subdirectory
        """
        tf.train.Saver(tf.trainable_variables()).restore(session, os.path.join(ckpts_dir, name, 'model.ckpt'))
