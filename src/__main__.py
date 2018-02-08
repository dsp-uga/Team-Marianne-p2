from pyspark import SparkContext, SparkConf
import argparse
import src.ByteFeature as src

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
    transform_x, transform_y = src.PreProcessor(sc).transform_data(train_x, byte_files_path, train_y)
    src.PreProcessor(sc).write_to_file(transform_x, transform_y)

    train_x, train_y = src.PreProcessor(sc).load_data(test_x, train_y)
    transform_x, transform_y = src.PreProcessor(sc).transform_data(train_x, byte_files_path, train_y)
    src.PreProcessor(sc).write_to_file(transform_x, transform_y)

    print('transformed data is: ', transform_x.collectAsMap())
    print('transformed label is: ', transform_y.collectAsMap())

def main():
    parser = argparse.ArgumentParser(description='Execute Commands')
    subcommands = parser.add_subparsers()

    # src preprocess <train_x> <train_y> <test_x> [<test_y>]
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
