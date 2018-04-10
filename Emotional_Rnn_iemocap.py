from Import_data import *
from tflearn.data_utils import to_categorical
from os.path import isfile
import tensorflow as tf


 #RNN Model Parameters
rnn_cell_size = 256 # hidden layer num of features (RNN hidden layer config size.)
rnn_data_classes = 4 # One value 
rnn_data_vec_size = 165
rnn_lstm_forget_bias = 1.0
rnn_dropout_keep_prob = 0.2
LEARNING_RATE = 0.0001
rnn_num_epochs = 6
batch_size = 0
dataset_size = 2200
##############################################################
####                 USEFUL FUNCTIONS                   ######
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

#function to split train_set into train and validation sets
def split_holdout_train_set(X_train_set,Y_train_set):

    hap, max_hap = 0, 262
    sad, max_sad = 0, 435
    neu, max_neu = 0, 496
    ang, max_ang = 0, 509

    new_X_train_set = []
    new_Y_train_set = []
    X_test_set = []
    Y_test_set = []

    print(X_train_set.shape)
    print(Y_train_set.shape)
    count = 0
    for x in Y_train_set:
        if x == 0:
            if hap < max_hap:
                new_X_train_set.append(X_train_set[count])
                new_Y_train_set.append(x)
                hap+=1
            else:
                Y_test_set.append(x)
                #Y_train_set = np.delete(Y_train_set,count)
                X_test_set.append(X_train_set[count])
                #X_train_set = np.delete(X_train_set,count)
        
        elif x == 1:
            if neu < max_neu:
                new_X_train_set.append(X_train_set[count])
                new_Y_train_set.append(x)
                neu+=1
            else:
                Y_test_set.append(x)
                #Y_train_set = np.delete(Y_train_set,count)
                X_test_set.append(X_train_set[count])
                #X_train_set = np.delete(X_train_set,count)
        
        elif x == 2:
            if sad < max_sad:
                new_X_train_set.append(X_train_set[count])
                new_Y_train_set.append(x)
                sad+=1
            else:
                Y_test_set.append(x)
                #Y_train_set = np.delete(Y_train_set,count)
                X_test_set.append(X_train_set[count])
                #X_train_set = np.delete(X_train_set,count)
        
        elif x == 3:
            if ang < max_ang:
                new_X_train_set.append(X_train_set[count])
                new_Y_train_set.append(x)
                ang+=1
            else:
                Y_test_set.append(x)
                #Y_train_set = np.delete(Y_train_set,count)
                X_test_set.append(X_train_set[count])
                #X_train_set = np.delete(X_train_set,count)
        count +=1
    
    return np.array(new_X_train_set), np.array(new_Y_train_set) , np.array(X_test_set) , np.array(Y_test_set)




#################################################
####    PREPARE TRAIN AND VALIDATION SETS    ####
#################################################

X_train_set, Y_train_set, max_frames_per_video_test = import_data('data/training20.csv')
X_train, Y_train, X_test, Y_test = split_holdout_train_set(X_train_set,Y_train_set)
max_frames_per_video_train = find_longest_sequence_lenght(X_train)
max_frames_per_video_test = find_longest_sequence_lenght(X_test)


trainX,lengths_vector = pad_sequences_with_zero_vectors(X_train,max_frames_per_video_train)
testX,lengths_vector_test = pad_sequences_with_zero_vectors(X_test,max_frames_per_video_test)
print(testX.shape)
trainY = to_categorical(Y_train,rnn_data_classes)
testY = to_categorical(Y_test,rnn_data_classes)
print(trainY.shape)



"""
X_test, Y_test,max_frames_per_video = import_data('data/training20.csv')
testX, lengths_vector_test = pad_sequences_with_zero_vectors(X_test,max_frames_per_video_test)
testY = to_categorical(Y_test,rnn_data_classes) """


###################################################
###### Prepare variables for dynamic rnn  #########
###################################################

graph = tf.Graph()

with graph.as_default():

    max_frames_per_video = tf.Variable( 0,name = "max_frames_per_video")

    inputs  = tf.placeholder(tf.float32, (None,None,rnn_data_vec_size),name= "inputs")  # (time, batch, in)

    outputs = tf.placeholder(tf.float32, (None, rnn_data_classes),name = "outputs") # (time, batch, out)

    Seq_length = tf.placeholder(tf.int32,name = "Seq_length")

    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_size, state_is_tuple=True,forget_bias= rnn_lstm_forget_bias,activation= tf.nn.tanh)

    cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = rnn_dropout_keep_prob)

    rnn_outputs, (rnn_states,last_timestep_output) = tf.nn.dynamic_rnn(cell_with_dropout, sequence_length=Seq_length, dtype=tf.float32, inputs=inputs)

    predicted_output = tf.contrib.layers.fully_connected(last_timestep_output,4,activation_fn=tf.nn.relu)

    #error = -(outputs * tf.log(predicted_output ) + (1.0 - outputs) * tf.log(1.0 - predicted_output  ))
    #error = tf.reduce_mean(error)

    outputs = tf.stop_gradient(outputs) # v2 performs backpropagation into labels too for adversial learning so we stop this behaviour
    error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=outputs, logits=predicted_output))
    

    #optimize
    train_fn = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(error)


    

    # Define the metric and update operations
    #a = tf.nn.softmax(predicted_output) # for debug
    #b= outputs                          # for debug
    
    accuracy, tf_metric_update = tf.metrics.accuracy(labels=tf.argmax(outputs,1),           #accuracy = tf.reduce_mean(tf.cast(tf.abs(outputs - predicted_output) < 0.5, tf.float32))
                                                        predictions=tf.argmax(predicted_output,1),
                                                        name="my_metric")

    # Evaluate model

    #correct_prediction = tf.equal(tf.argmax(outputs,1), tf.argmax(predicted_output,1)) #for debug
    #accu= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                      #for debug
    #acc =tf.reduce_mean(tf.to_float32(predictions == labels))                          #for debug

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
    saved_model = "checkpoints/model.ckpt"
    if tf.train.checkpoint_exists(saved_model):
        print("restoring model")
        saver.restore(session, saved_model)
    
    
    #session.run(tf.local_variables_initializer())

    if batch_size > 0:
        num_batches_per_epoch = int(dataset_size/batch_size) + 1

    print(testY)

    for epoch in range(rnn_num_epochs):
        
        if batch_size >0:

            session.run(running_vars_initializer)

            for batch_num in range(num_batches_per_epoch):

                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, dataset_size)

                X_train_batch = X_train[start_index:end_index]
                Y_train_batch = Y_train[start_index:end_index]

                max_frames_per_batch = find_longest_sequence_lenght(X_train_batch)

                train_batchX , lengths_vector_batch = pad_sequences_with_zero_vectors(X_train_batch,max_frames_per_batch)
                train_batchY = to_categorical(Y_train_batch,rnn_data_classes)

                epoch_error = session.run([error, train_fn,tf_metric_update], {
                    inputs: train_batchX,
                    outputs: train_batchY,
                    Seq_length:lengths_vector_batch
                })[0]

                valid_accuracy= session.run(accuracy,feed_dict= {
                inputs:  testX,
                outputs: testY,
                Seq_length:lengths_vector
                })
                print(valid_accuracy)
                
                print ("Epoch %d, Batch %d, train error: %.5f, valid accuracy: %.9f %%" % (epoch, batch_num, epoch_error, valid_accuracy ))
                save_path = saver.save(session, "checkpoint/model.ckpt")
                print("Model saved in path: %s" % save_path)

        else:
            session.run(running_vars_initializer)
            epoch_error = session.run([error, train_fn,tf_metric_update], {
                    inputs: trainX,
                    outputs: trainY,
                    Seq_length:lengths_vector,
                    max_frames_per_video: max_frames_per_video_train
            })[0]
            
            valid_accuracy= session.run(accuracy,feed_dict= {
                inputs:  testX,
                outputs: testY,
                Seq_length: lengths_vector_test,
                max_frames_per_video: max_frames_per_video_test
            })
 
           
            
            print ("Epoch %d, train error: %.5f, valid accuracy: %.9f %%" % (epoch, epoch_error, valid_accuracy ))
            
            save_path = saver.save(session, saved_model)
            print("Model saved in path: %s" % save_path)
