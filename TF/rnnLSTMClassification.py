import tensorflow as tf
# import data from dataset
from tensorflow.examples.tutorials.mnist import input_data
# load number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data/', one_hot='True')

BATCH_SIZE=100
N_STEPS=28
INPUTS_SIZE=28
LR=0.001
CELL_SIZE=128
TRAINING_ITERS=100000
N_CLASSES=10

class LSTMRNN(object):
   def __init__(self, n_steps, input_size, cell_size, batch_size,n_classes):
        self.n_steps = n_steps
        self.input_size = input_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        self.n_classes=n_classes
        with tf.name_scope('inputs'):
          #(100,28*28)
            self.xs = tf.placeholder(tf.float32, [None, n_steps*input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_classes], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
   def add_input_layer(self, ):
     l_in_x=tf.reshape(self.xs,[-1,self.n_steps],name='inputx')
     #(100*28,28)
     Ws_in=self._weight_variable([self.n_steps,self.cell_size])
     bs_in=self._bias_variable([self.cell_size,])
     l_in_y=tf.matmul(l_in_x,Ws_in)+bs_in
       #(100*28,128)
     self.l_in_y=tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='inputx2')
     #(100,28,128)
   def add_cell(self):
      lstm_cell=tf.contrib.rnn.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tuple=True)
      self.cell_init_state=lstm_cell.zero_state(self.batch_size,dtype=tf.float32)
      self.cell_outputs,self.cell_final_state=tf.nn.dynamic_rnn(
        lstm_cell,self.l_in_y,initial_state=self.cell_init_state,time_major=False
      )

   def add_output_layer(self):
     #(100)
      l_out_x=tf.unstack(tf.transpose(self.cell_outputs,[1,0,2]))
      Ws_out=self._weight_variable([self.cell_size,self.n_classes])
      bs_out=self._bias_variable([self.n_classes])
      self.pred=tf.matmul(l_out_x[-1],Ws_out)+bs_out
   def compute_cost(self):
      self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred,labels=self.ys))
      self.correct_pred=tf.equal(tf.argmax(self.pred,1),tf.arg_max(self.ys,1))
      self.accuracy=tf.reduce_mean(tf.cast(self.correct_pred,tf.float32))
   @staticmethod
   def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))
 
   def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1., )
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
 
   def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
 
if __name__ == '__main__':
     model = LSTMRNN(N_STEPS, INPUTS_SIZE, CELL_SIZE, BATCH_SIZE,N_CLASSES)
     sess = tf.Session()
     sess.run(tf.global_variables_initializer())

 
     for i in range(TRAINING_ITERS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)  # extract batch data
        if i == 0:
            feed_dict = {
                model.xs: batch_x,
                model.ys: batch_y,
            }
        else:
            feed_dict = {
                model.xs: batch_x,
                model.ys: batch_y,
                model.cell_init_state: state  
            }
 
        # train
        _, cost, state, pred,accuracy = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred,model.accuracy],
            feed_dict=feed_dict)
      
 
        # print result
        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            print('accuracy: ',accuracy)
            

    
