from Import_data import *
from lstm import *
import tensorflow as tf


 #RNN Model Parameters
rnn_cell_size = 256 # hidden layer num of features (RNN hidden layer config size. Start with 128)
rnn_data_classes = 4 # One value (vec negative, neutral, positive)
rnn_data_vec_size = 165
rnn_lstm_forget_bias = 1.0
rnn_dropout_keep_prob = 0.2
LEARNING_RATE = 0.001
rnn_num_epochs = 30
batch_size = 128
TINY = 1e-6 # to avoid NaNs in logs


#################################################
#### PREPARE TRAIN AND VALIDATION SETS###########
#################################################

X_train, Y_train, max_frames_per_video = import_data('training.csv')

pad = np.zeros(165)
lengths_vector = []
for x in X_train:
    lengths_vector.append(len(x))
    while len(x) < max_frames_per_video:
        x.append(np.array(pad))
trainX= np.array(list(X_train))
print(trainX.shape)

trainY = to_categorical(Y_train,rnn_data_classes)
print(trainY.shape)

#prepare validation data
X_test, Y_test,max_frames_per_video = import_data('training.csv')
pad = np.zeros(165)
lengths_vector = []

for x in X_test:
    lengths_vector.append(len(x))
    while len(x) < max_frames_per_video:
        x.append(np.array(pad))
testX= np.array(list(X_test))
print(trainX.shape)

testY = to_categorical(Y_test,rnn_data_classes)
print(trainY.shape)


###################################################
###### Prepare variables for dynamic rnn  #########
###################################################

graph = tf.Graph()

with graph.as_default():

    inputs  = tf.placeholder(tf.float32, (None,max_frames_per_video,rnn_data_vec_size),name= "inputs")  # (time, batch, in)

    outputs = tf.placeholder(tf.float32, (None, rnn_data_classes),name = "outputs") # (time, batch, out)

    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_size, state_is_tuple=True,forget_bias= rnn_lstm_forget_bias,activation= tf.nn.tanh)

    rnn_outputs, (rnn_states,last_timestep_output) = tf.nn.dynamic_rnn(cell, sequence_length=lengths_vector, dtype=tf.float32, inputs=inputs)

    predicted_output = tf.contrib.layers.fully_connected(last_timestep_output,4,activation_fn=tf.nn.softmax)

    #error = -(outputs * tf.log(predicted_output ) + (1.0 - outputs) * tf.log(1.0 - predicted_output  ))
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=predicted_output))
    #error = tf.reduce_mean(error)


    #optimize
    train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)


    

    # Define the metric and update operations
    a = tf.argmax(predicted_output,1) # for debug
    b= tf.argmax(outputs,1)#for debug
    
    accuracy, tf_metric_update = tf.metrics.accuracy(labels=tf.argmax(outputs,1),           #accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_output) < 0.5, tf.float32))
                                                        predictions=tf.argmax(predicted_output,1),
                                                        name="my_metric")

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)

    saver = tf.train.Saver()


################################################################################
##                           TRAINING LOOP                                    ##
################################################################################

with tf.Session(graph=graph) as session:
    
    session.run(tf.global_variables_initializer())

    saver.restore(session, "./model.ckpt")
    
    session.run(running_vars_initializer)
    #session.run(tf.local_variables_initializer())

    for epoch in range(rnn_num_epochs):
        
        epoch_error = 0
        epoch_error = session.run([error, train_fn], {
                inputs: trainX,
                outputs: trainY,
        })[0]
        
        valid_accuracy= session.run(tf_metric_update,feed_dict= {
            inputs:  testX,
            outputs: testY,
        })
        
        print ("Epoch %d, train error: %.5f, valid accuracy: %.9f %%" % (epoch, epoch_error, valid_accuracy ))
        save_path = saver.save(session, "./model.ckpt")
        print("Model saved in path: %s" % save_path)