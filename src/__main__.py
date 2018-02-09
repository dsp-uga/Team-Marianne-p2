from pyspark import SparkContext, SparkConf
import argparse
import src.byte_feature as src

def preprocess(args):
    '''Test and verify the preprocessing
    '''
    sc= SparkContext.getOrCreate()
    train_x = args.train_x
    train_y = args.train_y
    test_x = args.test_x
    test_y = args.test_y
    byte_files_path = args.byte_files_path

    train_x, train_y = src.PreProcessor(sc).load_data(train_x, train_y)
    training_data, training_labels = src.PreProcessor(sc).transform_data(train_x, byte_files_path, train_y)
    # This is just an example of how to write the file with file name
    src.PreProcessor(sc).write_to_file(training_data, 'training_data')

    test_x, test_y = src.PreProcessor(sc).load_data(test_x, test_y)
    testing_data, testing_labels = src.PreProcessor(sc).transform_data(test_x, byte_files_path, test_y)
    src.PreProcessor(sc).write_to_file(testing_data, 'testing_data')

    print('transformed data is: ', training_data.collectAsMap())
    print('transformed label is: ', training_labels.collectAsMap())

def main():
    parser = argparse.ArgumentParser(description='Execute Commands')
    subcommands = parser.add_subparsers()

    # src preprocess <train_x> <train_y> <test_x> [<test_y>] <byte_files_path>
    cmd = subcommands.add_parser('preprocess', description= 'Preprocess')
    cmd.add_argument('train_x', help='path of the training data')
    cmd.add_argument('train_y', help='path of the training labels')
    cmd.add_argument('test_x', help='path of thetesting data')
    cmd.add_argument('test_y', help='path of the testing labels', nargs='?', default=None)
    cmd.add_argument('byte_files_path', help='path of the byte files')
    cmd.set_defaults(func=preprocess)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
