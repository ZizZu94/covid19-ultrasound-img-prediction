import argparse
from datetime import datetime

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def parse_arguments():

    parser = argparse.ArgumentParser(description='Configurations.')
    parser.add_argument(
        '--model',
        default='resnet_18',
        type=str,
        help='Available models: efficient_net_b0 | efficient_net_b4 | resnet_18 | resnet_50')
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help='Leaning rate.')
    parser.add_argument(
        '--epochs',
        '-e',
        default=120,
        type=int,
        help='Number of training epochs.')
    parser.add_argument(
        '--batch_size',
        '-b',
        default=32, # 64
        type=int,
        help='Batch size.')
    parser.add_argument(
        '--num_workers',
        '-w',
        default=5,
        type=int,
        help='Number of workers in data loader')
    parser.add_argument(
        '--dropout',
        default=0.35,
        type=restricted_float,
        help='Dropout rate of the dropout layer')
    parser.add_argument(
        '--src_root',
        default='./src',
        type=str,
        help='Root folder for the src folder.')
    parser.add_argument(
        '--log_interval',
        default=50,
        type=int,
        help='Interval for printing')
    parser.add_argument(
        '--comment',
        '-c',
        default=datetime.now().strftime('%b%d_%H-%M-%S'),
        type=str,
        help='Comment to be appended to the model name to identify the run')
    parser.add_argument(
        '--model_name',
        '-n',
        default='covid19-resnet-18',
        type=str,
        help='Name of the model you want to use.')
    parser.add_argument(
        '--img_size',
        default=224,
        type=int,
        help='image size.')
    args = parser.parse_args()
    args.run_name = '-'.join([args.model_name, args.comment])

    return args