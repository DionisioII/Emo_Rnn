import tensorflow as tf
import sys


#writer = tf.summary.FileWriter("../checkpoint_1L_128/logs/metric_function_Rnn_1L_128")
writer = tf.summary.FileWriter("../checkpoint_1L_256/logs/metric_function_Rnn_1L_256")

loss = tf.Variable(1.56)
accuracy = tf.Variable(0.0)

tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)

tf.summary.merge_all()

fh = open('../checkpoint_1L_256/Rnn_1L_256cell.out')
count=0

while True:

    line= fh.readline()

    if line and line[0] == "E" :
        words = line.split()
        #print(words[4][:-1] + ' '+ words[7])
        loss = float(words[4][:-1])
        accuracy = float(words[7])
        global_step = int(words[1][:-1])
        summary = tf.Summary(value=[
        tf.Summary.Value(tag="loss", simple_value=loss),
        tf.Summary.Value(tag="accuracy", simple_value=accuracy),
        ])
        writer.add_summary(summary,count)
        count+=1
    if not line:
        break
fh.close()
#python3 ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir="../checkpointL_128/logs/metric_function_Rnn_1L_128"