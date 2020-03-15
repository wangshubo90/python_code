

writer = tf.summary.FileWriter(STORE_PATH, sess.graph)

# declare the training data placeholders
x = tf.placeholder(tf.float32, [None, 28, 28])
# reshape input x - for 28 x 28 pixels = 784
x_rs = tf.reshape(x, [-1, 784])
# scale the input data (maximum is 255.0, minimum is 0.0)
x_sc = tf.div(x_rs, 255.0)
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.int64, [None, 1])
# convert the y data to one hot values
y_one_hot = tf.reshape(tf.one_hot(y, 10), [-1, 10])

W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.01), name='W')
b1 = tf.Variable(tf.random_normal([300]), name='b')
hidden_logits = tf.add(tf.matmul(x_sc, W1), b1)
hidden_out = tf.nn.sigmoid(hidden_logits)

W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.05), name='W')
b2 = tf.Variable(tf.random_normal([10]), name='b')
logits = tf.add(tf.matmul(hidden_out, W2), b2)

# now let's define the cost function which we are going to train the model on
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot,
                                                            logits=logits))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_one_hot, 1), tf.argmax(logits, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(STORE_PATH, sess.graph)


with tf.name_scope("layer_1"):
    # now declare the weights connecting the input to the hidden layer
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.01), name='W')
    b1 = tf.Variable(tf.random_normal([300]), name='b')
    hidden_logits = tf.add(tf.matmul(x_sc, W1), b1)
    hidden_out = tf.nn.sigmoid(hidden_logits)

# add a summary to store the accuracy
tf.summary.scalar('acc_summary', accuracy)
merged = tf.summary.merge_all()

# start the session
with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(STORE_PATH, sess.graph)
    # initialise the variables
    total_batch = int(len(y_train) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y.reshape(-1, 1)})
            avg_cost += c / total_batch
        acc, summary = sess.run([accuracy, merged], feed_dict={x: x_test, y: y_test.reshape(-1, 1)})
        print("Epoch: {}, cost={:.3f}, test set accuracy={:.3f}%".format(epoch + 1, avg_cost, acc*100))
        writer.add_summary(summary, epoch)
print("\nTraining complete!")