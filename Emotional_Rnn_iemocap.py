from Import_data import *
from lstm import *
import tensorflow as tf


 #RNN Model Parameters
rnn_cell_size = 256 # hidden layer num of features (RNN hidden layer config size.)
rnn_data_classes = 4 # One value 
rnn_data_vec_size = 165
rnn_lstm_forget_bias = 1.0
rnn_dropout_keep_prob = 0.2
LEARNING_RATE = 0.001
rnn_num_epochs = 30
batch_size = 10
##############################################################
############# USEFUL FUNCTIONS################################
##############################################################


def find_longest_sequence_lenght(sequences):
    lenght= 0
    for x in sequences:
        if len(x)> lenght:
            lenght = len(x)
    return lenght

def pad_sequences_with_zero_vectors(sequences,max_sequence_num):
    pad = np.zeros(rnn_data_vec_size)
    lengths_vector = []
    for x in sequences:
        lengths_vector.append(len(x))
        while len(x) < max_sequence_num:
            x.append(np.array(pad))
    return np.array(list(sequences)),lengths_vector


#################################################
#### PREPARE TRAIN AND VALIDATION SETS###########
#################################################

X_train, Y_train, max_frames_per_video_test = import_data('training.csv')

trainX,lengths_vector = pad_sequences_with_zero_vectors(X_train,max_frames_per_video_test)
print(trainX.shape)
trainY = to_categorical(Y_train,rnn_data_classes)
print(trainY.shape)

X_test, Y_test,max_frames_per_video = import_data('training.csv')
testX, lengths_vector_test = pad_sequences_with_zero_vectors(X_test,max_frames_per_video_test)
testY = to_categorical(Y_test,rnn_data_classes)


###################################################
###### Prepare variables for dynamic rnn  #########
###################################################

graph = tf.Graph()

with graph.as_default():

    inputs  = tf.placeholder(tf.float32, (None,max_frames_per_video,rnn_data_vec_size),name= "inputs")  # (time, batch, in)

    outputs = tf.placeholder(tf.float32, (None, rnn_data_classes),name = "outputs") # (time, batch, out)

    Seq_length = tf.placeholder(tf.int32,name = "Seq_length")

    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_size, state_is_tuple=True,forget_bias= rnn_lstm_forget_bias,activation= tf.nn.tanh)

    rnn_outputs, (rnn_states,last_timestep_output) = tf.nn.dynamic_rnn(cell, sequence_length=Seq_length, dtype=tf.float32, inputs=inputs)

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

    # Evaluate model

    correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(predicted_output,1))
    accu= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #acc =tf.reduce_mean(tf.to_float32(predictions == labels))

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

    if batch_size > 0:
        num_batches_per_epoch = int(310/batch_size) + 1


    for epoch in range(rnn_num_epochs):
        
        if batch_size >0:

            for batch_num in range(num_batches_per_epoch):
                
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, 310)

                X_train_batch = X_train[start_index:end_index]
                Y_train_batch = Y_train[start_index:end_index]

                max_frames_per_batch = find_longest_sequence_lenght(X_train_batch)

                train_batchX , lengths_vector_batch = pad_sequences_with_zero_vectors(X_train_batch,max_frames_per_batch)
                train_batchY = to_categorical(Y_train_batch,rnn_data_classes)

                epoch_error = session.run([error, train_fn], {
                    inputs: train_batchX,
                    outputs: train_batchY,
                    Seq_length:lengths_vector_batch
                })[0]

                valid_accuracy= session.run(accu,feed_dict= {
                inputs:  testX,
                outputs: testY,
                Seq_length:lengths_vector
                })
                print(valid_accuracy)
                #print(valid_accuracy.eval())
                print ("Epoch %d, Batch %d, train error: %.5f, valid accuracy: %.9f %%" % (epoch, batch_num, epoch_error, valid_accuracy ))
                save_path = saver.save(session, "./model.ckpt")
                print("Model saved in path: %s" % save_path)

        else:

            epoch_error = session.run([error, train_fn,tf_metric_update], {
                    inputs: trainX,
                    outputs: trainY,
                    Seq_length:lengths_vector
            })[0]
            
            valid_accuracy= session.run(accu,feed_dict= {
                inputs:  testX,
                outputs: testY,
                Seq_length: lengths_vector_test
            })
            
            print ("Epoch %d, train error: %.5f, valid accuracy: %.9f %%" % (epoch, epoch_error, valid_accuracy ))
            save_path = saver.save(session, "./model.ckpt")
            print("Model saved in path: %s" % save_path)
