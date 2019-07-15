from utils import create_seq_generator
import model
import argparse


def learning(args):
    # Create the generator objects
    my_training_batch_generator, num_training_samples = create_seq_generator(args.train_dir, args.batch_size,
                                                                             extA='1.jpg', extB='2.jpg', extL='re.txt')
    my_val_batch_generator, num_validation_samples = create_seq_generator(args.val_dir, args.batch_size, extA='1.jpg',
                                                                          extB='2.jpg', extL='re.txt')
    my_test_batch_generator, num_test_samples = create_seq_generator(args.test_dir, args.batch_size, extA='1.jpg',
                                                                     extB='2.jpg', extL='re.txt')

    hnn = model.HomographyNN(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
                             momentum=args.momentum, weights_name=args.weights)

    hnn.generate_homography_nn_sgd()

    # Train the CNN
    hnn.fit(training_generator=my_training_batch_generator, dimension_train=num_training_samples,
            val_generator=my_val_batch_generator, dimension_val=num_validation_samples)

    hnn.save_weights(args.weights)

    # Test the model
    loss = hnn.test_generator(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    print('Test loss: {}'.format(loss))
    return loss


def evaluate(args):
    # Create the generator object
    my_test_batch_generator, num_test_samples = create_seq_generator(args.test_dir, args.batch_size)

    # Build the CNN
    hnn = model.HomographyNN(batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.lr,
                             momentum=args.momentum)
    hnn.generate_homography_nn_sgd()

    # Load the weights
    hnn.load_weights(args.weights)

    # Perform inference
    loss = hnn.test_generator(test_generator=my_test_batch_generator, dimension_test=num_test_samples)
    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str, default='../train', help='Training directory.')
    parser.add_argument('--test_dir', type=str, default='../test', help='Test directory.')
    parser.add_argument('--val_dir', type=str, default='../val', help='Validation directory.')

    parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate, default: 5e-3')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum, default: 0.9')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size, default: 64')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train, default: 18')

    parser.add_argument('--weights', type=str, default='re_12ep_mse',
                        help='Name of the file in which to save the weights.')
    args = parser.parse_args()
    test_loss = learning(args)
    # test_loss = evaluate(args)
    print('Test loss: {}'.format(test_loss))
