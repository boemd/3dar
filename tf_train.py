import tensorflow as tf
from tf_model import HomographyNet
from datetime import datetime
from tf_data_reader import get_dataset
import os
import logging

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 16,
                        'batch size, default: 1')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_string('X', 'data/tfrecords/train.tfrecords',
                       'X tfrecords file for training, default: data/tfrecords/train_x.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('max_iter', 200000,
                        'maximum number of iterations during training, default: 400000')

def train():

    if FLAGS.load_model is not None:
        # load the specified model
        checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    else:
        # create checkpoint directory
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "checkpoints/{}".format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error:
            pass

    write_config_file(checkpoints_dir)
    graph = tf.Graph()
    with graph.as_default():
        net = HomographyNet(
             train_file=FLAGS.X,
             batch_size=FLAGS.batch_size,
             learning_rate=FLAGS.learning_rate,
             )
        logging.info('HomographyNet initialized')
        loss = net.model()
        x_train, y_train, _ = get_dataset(FLAGS.X, FLAGS.batch_size)
        optimizer = net.optimize(loss)
        logging.info('Optimizer OK')
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:

            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('# Total parameters of the network: ', total_parameters, '#')

            while (not coord.should_stop()) and (step < FLAGS.max_iter):

                # train
                x_train_val, y_train_val = sess.run(
                    [x_train, y_train],
                    feed_dict={
                        net.x_train: x_train_val,
                        net.y_train: y_train_val
                    })
                _, loss_val, summary = sess.run([optimizer, loss, summary_op])

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 1000 == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  loss   : {}'.format(loss_val))

                if step % 10000 == 0:
                    save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step, write_meta_graph=False)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def write_config_file(checkpoints_dir):
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    with open(checkpoints_dir + '/config.txt', 'w') as c:
        c.write(date_time + '\n')
        c.write('Batch size:' + str(FLAGS.batch_size) + '\n')
        c.write('Iterations:' + str(FLAGS.max_iter) + '\n')


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
