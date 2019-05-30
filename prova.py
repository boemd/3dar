
import generator
import tensorflow as tf
from utils import create_generator
import model
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'C:/Users/gabro/Documents/3D/ms-coco/train', 'Input directory')
tf.flags.DEFINE_string('test_dir', 'C:/Users/gabro/Documents/3D/ms-coco/test', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'C:/Users/gabro/Documents/3D/ms-coco/val', 'Validation directory')


def learning(batch_size, epochs, lr, momentum):

    # Create the generator objects
    my_training_batch_generator, num_training_samples = create_generator(FLAGS.input_dir, batch_size)
    my_val_batch_generator, num_validation_samples = create_generator(FLAGS.val_dir, batch_size)
    my_test_batch_generator, num_test_samples = create_generator(FLAGS.test_dir, batch_size)

    hnn = model.HomographyNN(batch_size=batch_size, epochs=epochs, learning_rate=lr, momentum=momentum)
    hnn.generate_homography_nn()
    hnn.fit(training_generator=my_training_batch_generator, dimension_train=num_training_samples,
            val_generator=my_val_batch_generator, dimension_val=num_validation_samples)
    hnn.save_weights("64batch")

    # Test the model
    [loss, accuracy] = hnn.test_generator(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    return accuracy


if __name__ == '__main__':
    lr = 0.005
    momentum = 0.9
    a = learning(32, 8, lr, momentum)
    print(a)



    #datagen = ImageDataGenerator() 6
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(FLAGS.input_dir, class_mode=None, batch_size=64)
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(FLAGS.val_dir, class_mode=None, batch_size=64)
    # # load and iterate test dataset
    # test_it = datagen.flow_from_directory(FLAGS.test_dir, class_mode='binary', batch_size=64)
    # model = build_model()
    # num_training_samples = len(a_train)