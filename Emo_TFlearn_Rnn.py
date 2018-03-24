from Import_data import *
from lstm import *
from tflearn.data_utils import to_categorical, pad_sequences
import tensorflow as tf



X_train, Y_train, max_frames_per_video = import_data()

print(max_frames_per_video)


 #Data preprocessing
# NOTE: Padding is required for dimension consistency. This will pad sequences
# with 0 at the end, until it reaches the max sequence length. 0 is used as a
# masking value by dynamic RNNs in TFLearn; a sequence length will be
# retrieved by counting non zero elements in a sequence. Then dynamic RNN step
# computation is performed according to that length.
#print(X_train)

pad = np.zeros(165)
for x in X_train:
    while len(x) < max_frames_per_video:
        x.append(np.array(pad))
#sequences = [np.pad(x, (29 - len(x), 0), 'constant', constant_values = (0.,0.)) for x in X_train]
        
#trainX = pad_sequences(X_train,maxlen=max_frames_per_video,value=0.)
#print(len(newTrain[0]))
trainX= np.array(list(X_train))
print(trainX.shape)


#np.nan_to_num(trainX)


#trainX=trainX.reshape(-1,310,29,165)
#testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors

trainY = to_categorical(Y_train,4)

#testY = to_categorical(testY)
input_x = tf.placeholder(tf.float64, [None, max_frames_per_video], name="input_x")
# Network building
net = tflearn.input_data(shape=[None,max_frames_per_video,165])
# Masking is not required for embedding, sequence length is computed prior to
# the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 512, dropout=0.2,dynamic=True)#return_seq=True)
#net = tflearn.lstm(net, 128)
net = tflearn.fully_connected(net, 4, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('checkpoints/rnn.tflearn')
model.fit(trainX, trainY,  show_metric=True,
batch_size=100,n_epoch=10)

model.save('checkpoints/rnn.tflearn')


