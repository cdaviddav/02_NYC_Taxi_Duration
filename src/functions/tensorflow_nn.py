import tensorflow as tf
import datetime
import os
from settings import Settings
settings = Settings()




def run_nn(x_train, y_train, x_test, y_test, learning_rate, training_epochs, batch_size, display_step, n_hidden_1,
           n_hidden_2, n_hidden_3, n_hidden_4, n_classes):

    n_input = x_train.shape[1]
    total_len = x_train.shape[0]

    with tf.name_scope('input'):
        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, n_input], name='Features')
        y = tf.placeholder(tf.float32, [None, n_classes], name='LabelData')


    def weight_variable(shape):
        # initial = tf.contrib.layers.xavier_initializer(shape)
        initial = tf.random_normal(shape, 0, 0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.zeros(shape)
        # initial = tf.random_normal(shape, 0, 0,1)
        return tf.Variable(initial)


    def variable_summaries(var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


    def nn_layer(input_tensor, input_dim, output_dim, layer_name, output_layer):

        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)

            if output_layer is True:
                activations = preactivate
            else:
                activations = tf.nn.relu(preactivate, name='activation')

            tf.summary.histogram('activations', activations)
            return activations

    hidden1 = nn_layer(x, n_input, n_hidden_1, 'layer1', output_layer=False)
    hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2', output_layer=False)
    hidden3 = nn_layer(hidden2, n_hidden_2, n_hidden_3, 'layer3', output_layer=False)
    hidden4 = nn_layer(hidden3, n_hidden_3, n_hidden_4, 'layer4', output_layer=False)
    hidden5 = nn_layer(hidden4, n_hidden_4, n_classes, 'layer5', output_layer=True)
    pred = hidden5


    with tf.name_scope('Loss'):
        # Softmax Cross entropy (cost function)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        loss = tf.reduce_mean(tf.square(pred - y))

    with tf.name_scope('TRAIN'):
        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # Op to calculate every variable gradient
        grads = tf.gradients(loss, tf.trainable_variables())
        grads = list(zip(grads, tf.trainable_variables()))
        # Op to update all variables according to their gradient
        apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
        optimizer = optimizer.minimize(loss)

    with tf.name_scope('Accuracy'):
        # Accuracy
        acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Create a summary to monitor cost tensor
    tf.summary.scalar("loss", loss)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("accuracy", acc)
    # Create summaries to visualize weights
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    # Summarize all gradients
    for grad, var in grads:
        tf.summary.histogram(var.name + '/gradient', grad)
    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(
            os.path.join(settings.tensorboard, str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))),
            graph=tf.get_default_graph())

        for epoch in range(training_epochs):

            avg_cost = 0.
            avg_acc = 0.
            total_batch = int(total_len / batch_size)
            # Loop over all batches
            for i in range(total_batch - 1):
                batch_x = x_train[i * batch_size:(i + 1) * batch_size]
                batch_y = y_train[i * batch_size:(i + 1) * batch_size]

                _, c, a, summary = sess.run([apply_grads, loss, acc, merged_summary_op], feed_dict={x: batch_x, y: batch_y})

                summary_writer.add_summary(summary, epoch * total_batch + i)
                avg_cost += c / total_batch
                avg_acc += a / total_batch

            # Display logs after every 10 epochs
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "acc=", "{:.9f}".format(avg_acc), "cost=", "{:.9f}".format(avg_cost))

        print("Optimization Finished!")

        # Test model
        print("Accuracy:", acc.eval({x: x_test, y: y_test}))

    # tensorboard --logdir=tensorboard/ --host localhost --port 8088