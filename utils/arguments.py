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
        default=64, # 64
        type=int,
        help='Batch size.')
    parser.add_argument(
        '--sensors',
        nargs='+',
        default=['linear', 'convex', 'unknown'],
        help='Sensors to be used.')
    parser.add_argument(
        '--num_workers',
        '-w',
        default=5,
        type=int,
        help='Number of workers in data loader')
    parser.add_argument(
        '--dropout',
        default=0.4,
        type=restricted_float,
        help='Dropout rate of the dropout layer')
    parser.add_argument(
        '--dataset_root',
        default='./data',
        type=str,
        help='Root folder for the datasets.')
    parser.add_argument(
        '--src_root',
        default='/home/zihadul.azam/ultra/src',
        #default='/mnt/d/Uni/Masters/MID/thesis-project/covid-19-frame-pred',
        type=str,
        help='Root folder for the src folder.')
    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='Random seed.')
    parser.add_argument(
        '--split',
        default='patient_hash',
        type=str,
        help='The split strategy.')
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
        default='covid19',
        type=str,
        help='Name of the model you want to use.')
    parser.add_argument(
        '--img_size',
        default=224,
        type=int,
        help='image size.')
    parser.add_argument(
        '--fixed_scale',
        default=False,
        action='store_true',
        help='Use fixed scaling for the STN.')
    parser.add_argument(
        '--lambda_cons',
        type=float,
        default=1.,
        help='weight for consistency regularization loss')
    parser.add_argument(
        '--lambda_stn_params',
        type=float,
        default=1.,
        help='weight for the scaling params loss')
    parser.add_argument(
        '--multiplier',
        default=2,
        type=int,
        help='multiplier for sord loss')
    args = parser.parse_args()
    args.run_name = '-'.join([args.model_name, args.comment])

    return args