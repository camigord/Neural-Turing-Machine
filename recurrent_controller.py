import tensorflow as tf
from controller import BaseController

class RecurrentController(BaseController):
    def network_vars(self):
        self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100)
        self.state = tf.Variable(tf.zeros([self.batch_size, 100]), trainable=False)
        self.output = tf.Variable(tf.zeros([self.batch_size, 100]), trainable=False)

    def network_op(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def update_state(self, new_state):
        return tf.group(
            self.output.assign(new_state[0]),
            self.state.assign(new_state[1])
        )

    def get_state(self):
        return (self.output, self.state)
