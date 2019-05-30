import tensorflow as tf
from hnet import HomographyNet00
from tf_data_reader import get_dataset


class HomographyNet:
    def __init__(self,
                 train_file='',
                 batch_size=1,
                 learning_rate=2e-4,
                 ):
        self.train_file = train_file
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        self.x_train = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 2], name='x_train')
        self.y_train = tf.placeholder(tf.float32, shape=[self.batch_size, None, None, 2], name='y_train')

        self.H = HomographyNet00('net0', self.is_training)

    def model(self):

        #x_train, y_train, _ = get_dataset(self.train_file, self.batch_size)

        y_pred = self.H(self.x_train)

        loss = tf.losses.mean_squared_error(self.y_train, y_pred)

        tf.summary.scalar('loss/mse', loss)

        return loss

    def optimize(self, loss):
        def make_optimizer(t_loss, variables, name='SGD'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            start_decay_step = 40000
            decay_steps = 40000
            decay_rate = 0.5
            learning_rate = (
                tf.where(
                    condition=tf.greater_equal(global_step, start_decay_step),
                    x=tf.train.exponential_decay(learning_rate=starter_learning_rate, global_step=global_step,
                                                 decay_steps=decay_steps, decay_rate=decay_rate, staircase=True),
                    y=starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.GradientDescentOptimizer(learning_rate, name=name)
                  .minimize(t_loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        optimizer = make_optimizer(loss, self.H.variables, name='SGD')

        with tf.control_dependencies([optimizer]):
            return tf.no_op(name='optimizer')
