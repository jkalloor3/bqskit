"""
params.py
This module contains all parameter handling for the project, including
all command line tunable parameter definitions, any preprocessing
of those parameters, or generation/specification of other parameters.
"""

import argparse
import os
import yaml
from bqskit.runtime import default_server_port

def make_argparser():
    parser = argparse.ArgumentParser(description='Arguments for performance meassurment')   

    
    parser.add_argument('--seed', type=int)
    
    parser.add_argument('--input_qasm', type=str, required=True)
    
    
    parser.add_argument('--print_amount_of_nodes', type=int,  default=1)
    parser.add_argument('--amount_of_workers', type=int, default=-1)
    parser.add_argument('--amount_gpus_per_node', type=int, default=0)

    parser.add_argument('--partitions_size', type=int,  required=True)
    parser.add_argument('--perform_while', action='store_true')
    

    parser.add_argument('--use_detached', action='store_true')
    parser.add_argument('--detached_server_ip', type=str, default='localhost')
    parser.add_argument('--detached_server_port', type=int, default=default_server_port)


    parser.add_argument('--instantiator', type=str, choices=['CERES', 'QFACTOR-RUST', 'QFACTOR-JAX', 'LBFGS', 'CERES_P', 'QFACTOR-RUST_P',  'LBFGS_P'], required=True)
    parser.add_argument('--multistarts', type=int, default=16)
    parser.add_argument('--max_iters', type=int,  default=10000000)
    parser.add_argument('--min_iters', type=int,  default=0)
    parser.add_argument('--diff_tol_r', type=float,  default=1e-5)
    parser.add_argument('--diff_tol_a', type=float,  default=1e-10)
    parser.add_argument('--dist_tol', type=float,  default=1e-10) 
    
    
    parser.add_argument('--diff_tol_step_r', type=float,  default=0.1) 
    parser.add_argument('--diff_tol_step', type=int,  default=100) 
    parser.add_argument('--beta', type=float,  default=0.0) 



    return parser.parse_args()

def get_params():
    parser = make_argparser()
    print(parser)

    return parser
