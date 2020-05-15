# Functions shared by multiple scripts
import argparse
import json
from warnings import warn
import datetime
import os
import torch


def get_config(config_parser, sys_parser=argparse.ArgumentParser(add_help=False), generate_output_folder=False,
               description=None):
    # Read config file
    file_parser = argparse.ArgumentParser(add_help=False)
    file_parser.add_argument('-c', '--config', default='',
                             help='path to json config file that overide the default options.'
                                  ' Commandline options have precedence over the options in this '
                                  'config file. (default: empty)')
    src, rem_args = file_parser.parse_known_args()
    # Override defaults from config file (if this is not empty)
    if src.config:
        with open(src.config, 'r') as f:
            config_dict = json.load(f)
        config_parser.set_defaults(**config_dict)
    # Read parameters from line command (overriding config file)
    config, rem_args = config_parser.parse_known_args(rem_args)
    # Add cuda option to system parser
    sys_parser.add_argument('--cuda', action='store_true',
                            help='use cuda for computations. (default: False)')
    # Add folder option to
    if generate_output_folder:
        default_out_folder = os.path.join(os.getcwd(), 'output-' + str(datetime.datetime.now()).replace(":", "_").replace(" ", "_"))
        sys_parser.add_argument('--folder', default=default_out_folder,
                                help='output folder(default: ./output-YYYY-MM-DD HH:MM:SS.MMMMMM)')
    else:
        sys_parser.add_argument('--folder', type=str, default='',
                                help='Folder to use as input (required)')
    args, unk = sys_parser.parse_known_args(rem_args)
    # Generate output folder if needed
    if generate_output_folder and not os.path.exists(args.folder):
        os.makedirs(args.folder)
    # Save config file
    with open(os.path.join(args.folder, config_parser.prog+'config.json'), 'w') as f:
        json.dump(vars(config), f, indent='\t')

    #  Final parser is needed for generating help documentation
    parser = argparse.ArgumentParser(parents=[sys_parser, config_parser, file_parser], description=description)
    _, unk = parser.parse_known_args(unk)
    # Check for unknown options
    if unk:
        warn("Unknown arguments:" + str(unk) + ".")
    # Set device
    device = torch.device('cuda:0' if args.cuda else 'cpu')
    return args, config, device, args.folder

