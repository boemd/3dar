
import tensorflow as tf
from utils import create_seq_generator
import model
FLAGS = tf.flags.FLAGS


tf.flags.DEFINE_string('train_dir', 'D:/tesisti/Boem/ar3/train', 'Training directory')
tf.flags.DEFINE_string('test_dir', 'D:/tesisti/Boem/ar3/test', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'D:/tesisti/Boem/ar3/val', 'Validation directory')
'''
tf.flags.DEFINE_string('train_dir', 'D:/tesisti/Boem/ar2/val_set/val_set', 'Training directory')
tf.flags.DEFINE_string('test_dir', 'D:/tesisti/Boem/ar2/test_set/test_set', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'D:/tesisti/Boem/ar2/val_set/val_set', 'Validation directory')
'''

tf.flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate, default: 5e-3')
tf.flags.DEFINE_float('momentum', 0.5, 'Momentum, default: 0.9')
tf.flags.DEFINE_integer('batch_size', 64, 'Batch size, default: 64')
tf.flags.DEFINE_integer('epochs', 12, 'Number of epochs to train, default: 18')

tf.flags.DEFINE_string('save_weights', 're_12ep_mse', 'name of the file in which to save')


def learning(batch_size, epochs, lr, momentum):

    # Create the generator objects
    my_training_batch_generator, num_training_samples = create_seq_generator(FLAGS.train_dir, batch_size, extA='1.jpg', extB='2.jpg', extL='re.txt')
    my_val_batch_generator, num_validation_samples = create_seq_generator(FLAGS.val_dir, batch_size, extA='1.jpg', extB='2.jpg', extL='re.txt')
    my_test_batch_generator, num_test_samples = create_seq_generator(FLAGS.test_dir, batch_size, extA='1.jpg', extB='2.jpg', extL='re.txt')

    hnn = model.HomographyNN(batch_size=batch_size, epochs=epochs, learning_rate=lr, momentum=momentum,
                             weights_name=FLAGS.weights)
    hnn.generate_homography_nn_sgd()
    hnn.fit(training_generator=my_training_batch_generator, dimension_train=num_training_samples,
            val_generator=my_val_batch_generator, dimension_val=num_validation_samples)
    hnn.save_weights(FLAGS.save_weights)

    # Test the model
    loss = hnn.test_generator(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    print('Test loss: {}'.format(loss))
    return loss


def evaluate(model_weights, batch_size, epochs, lr, momentum):
    my_test_batch_generator, num_test_samples = create_seq_generator(FLAGS.test_dir, batch_size)

    hnn = model.HomographyNN(batch_size=batch_size, epochs=epochs, learning_rate=lr, momentum=momentum)
    hnn.generate_homography_nn_sgd()
    hnn.load_weights(model_weights)
    loss = hnn.test_generator(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    return loss


if __name__ == '__main__':
    loss = learning(FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate, FLAGS.momentum)
    # loss = evaluate('64batch_18epochs_mse_sgd_1', FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate, FLAGS.momentum)
    print('Test loss: {}'.format(loss))




    #datagen = ImageDataGenerator() 6
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(FLAGS.input_dir, class_mode=None, batch_size=64)
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(FLAGS.val_dir, class_mode=None, batch_size=64)
    # # load and iterate test dataset
    # test_it = datagen.flow_from_directory(FLAGS.test_dir, class_mode='binary', batch_size=64)
    # model = build_model()
    # num_training_samples = len(a_train)