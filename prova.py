
import MY_Generator
import tensorflow as tf
import usefull_function
import generate_model
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('input_dir', 'D:/dataset/train_set/train_set', 'Input directory')
tf.flags.DEFINE_string('test_dir', 'D:/dataset/test_set', 'Test directory')
tf.flags.DEFINE_string('val_dir', 'D:/dataset/val_set', 'Validation directory')



def learning(batch_size, epochs, lr, momentum):

    # Load the File in the PATH
    a_train, b_train, mat_train = usefull_function.data_reader(FLAGS.input_dir)
    a_val, b_val, mat_val = usefull_function.data_reader(FLAGS.val_dir)
    a_test, b_test, mat_test = usefull_function.data_reader(FLAGS.test_dir)
    num_training_samples = len(a_train)
    num_validation_samples = len(a_val)
    num_test_samples = len(a_test)

    # Create the generator objects
    my_training_batch_generator = MY_Generator.Generator(a_train, b_train, mat_train, batch_size)
    my_val_batch_generator = MY_Generator.Generator(a_val, b_val, mat_val, batch_size)
    my_test_batch_generator = MY_Generator.Generator(a_test, b_test, mat_test, batch_size)

    hnn = generate_model.HomographyNN(batch_size, epochs)
    hnn.generate_Homography_NN(lr=lr, momentum=momentum)

    hnn.load_weights("batch128")

    # Test the model
    [loss, mtr] = hnn.test(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    return loss, mtr


if __name__ == '__main__':
    lr = 0.005
    momentum = 0.9
    loss, mtr = learning(128, 1, lr, momentum)
    print(loss)
    print(mtr)




    #datagen = ImageDataGenerator()
    # # load and iterate training dataset
    # train_it = datagen.flow_from_directory(FLAGS.input_dir, class_mode=None, batch_size=64)
    # # load and iterate validation dataset
    # val_it = datagen.flow_from_directory(FLAGS.val_dir, class_mode=None, batch_size=64)
    # # load and iterate test dataset
    # test_it = datagen.flow_from_directory(FLAGS.test_dir, class_mode='binary', batch_size=64)
    # model = build_model()
    # num_training_samples = len(a_train)