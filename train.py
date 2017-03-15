import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import getopt
import sys
import os

from ntm import NTM
from feedforward_controller import FeedforwardController

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size):
    input_data = np.zeros((batch_size, 2 * length + 2, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 2, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 2))
    input_data[:, 0, 0] = 1
    input_data[:, 1:length+1, 1:size-1] = sequence
    input_data[:, length+1, -1] = 1  # the end symbol
    target_output[:, length + 2:, 1:size-1] = sequence

    return input_data, target_output

'''def generate_data(batch_size, length, size):
    input_data = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)
    target_output = np.zeros((batch_size, 2 * length + 1, size), dtype=np.float32)

    sequence = np.random.binomial(1, 0.5, (batch_size, length, size - 1))

    input_data[:, :length, :size - 1] = sequence
    input_data[:, length, -1] = 1  # the end symbol
    target_output[:, length + 1:, :size - 1] = sequence

    return input_data, target_output
'''

def binary_cross_entropy(predictions, targets):
    return tf.reduce_mean(-1 * targets * tf.log(predictions) - (1 - targets) * tf.log(1 - predictions))


if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname , 'checkpoints')
    tb_logs_dir = os.path.join(dirname, 'logs')

    batch_size = 1
    input_size = output_size = 10
    sequence_max_length = 10
    memory_size = 128
    word_size = 20
    read_heads = 1
    shift_range = 1

    learning_rate = 1e-4
    momentum = 0.9
    decay_rate = 0.95

    from_checkpoint = None
    iterations = 100000

    options,_ = getopt.getopt(sys.argv[1:], '', ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    graph = tf.Graph()

    with graph.as_default():
        with tf.Session(graph=graph) as session:

            llprint("Building Computational Graph ... ")

            with tf.name_scope("Optimizer") as optimizer_scope:
                optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=decay_rate,momentum=momentum)

            turing_machine = NTM(
                FeedforwardController,
                input_size,
                output_size,
                memory_size,
                word_size,
                read_heads,
                shift_range,
                batch_size
            )

            with tf.name_scope("Loss") as loss_scope:
                # squash the DNC output between 0 and 1
                output, _ = turing_machine.get_outputs()
                squashed_output = tf.clip_by_value(tf.sigmoid(output), 1e-6, 1. - 1e-6)
                loss = binary_cross_entropy(squashed_output, turing_machine.target_output)

            summaries = []

            with tf.name_scope(optimizer_scope):
                gradients = optimizer.compute_gradients(loss)
                for i, (grad, var) in enumerate(gradients):
                    if grad is not None:
                        summaries.append(tf.summary.histogram(var.name + '/grad', grad))
                        gradients[i] = (tf.clip_by_value(grad, -10, 10), var)

                apply_gradients = optimizer.apply_gradients(gradients)

            with tf.name_scope(loss_scope):
                summaries.append(tf.summary.scalar("Loss", loss))
                summarize_op = tf.summary.merge(summaries)
                no_summarize = tf.no_op()
                summarizer = tf.summary.FileWriter(tb_logs_dir, session.graph)

            llprint("Done!\n")

            llprint("Initializing Variables ... ")
            session.run(tf.global_variables_initializer())
            llprint("Done!\n")

            if from_checkpoint is not None:
                llprint("Restoring Checkpoint %s ... " % (from_checkpoint))
                turing_machine.restore(session, ckpts_dir, from_checkpoint)
                llprint("Done!\n")


            last_100_losses = []

            for i in xrange(iterations + 1):
                llprint("\rIteration %d/%d" % (i, iterations))

                random_length = np.random.randint(1, sequence_max_length + 1)
                input_data, target_output = generate_data(batch_size, random_length, input_size)

                summarize = (i % 100 == 0)
                take_checkpoint = ((i != 0) and (i % 1000 == 0)) or (i % iterations == 0)

                loss_value, _, summary = session.run([
                    loss,
                    apply_gradients,
                    summarize_op if summarize else no_summarize
                ], feed_dict={
                    turing_machine.input_data: input_data,
                    turing_machine.target_output: target_output,
                    turing_machine.sequence_length: 2 * random_length + 2
                })

                last_100_losses.append(loss_value)
                summarizer.add_summary(summary, i)

                if summarize:
                    llprint("\n\tAvg. Logistic Loss: %.4f\n" % (np.mean(last_100_losses)))
                    last_100_losses = []

                if take_checkpoint:
                    llprint("\nSaving Checkpoint ... "),
                    turing_machine.save(session, ckpts_dir, 'step-%d' % (i))
                    llprint("Done!\n")
