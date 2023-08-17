import torch
import numpy as np

from polygon import get_polygon

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import RNN

from visualize import compute_ratemaps

import json

import os
import argparse

def generate_options( parameters ):

    # directories

    load_path = os.getcwd() + '/pre_computed/'
    save_path = os.getcwd() + '/output/'

    parser = argparse.ArgumentParser()

    global_parameters = parameters.keys()

    for p in global_parameters:
        try:
            for k, v in parameters[p].items():

                parser.add_argument(
                    '--' + k,
                    default=v,
                    help=f'{k} parameter'
                )
        except:
            parser.add_argument(
                '--' + p,
                default=parameters[p],
                help=f'{p} parameter'
            )

    # load and save directories

    parser.add_argument(
        '--load_path',
        default=load_path,
        help='directory to load example models'
    )

    parser.add_argument(
        '--save_path',
        default=save_path,
        help='directory to save trained models'
    )
    
    # device

    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='device to use for training'
    ) 

    return parser.parse_args()

if __name__ == '__main__':

    # where am i?
    PATH = os.getcwd() + '/'
    precomputed_path = PATH + 'precomputed/'

    input_file = PATH + 'experiments/cube_10_10_test.json'

    print(f'PATH: {PATH}')
    print(f'experiment: {input_file}')

    # load JSON file
    with open( input_file ) as f:
        parameters = json.load( f )

    # generate options
    options = generate_options( parameters )

    # get polygon
    polygon = get_polygon( options.shape )

    # place cells
    place_cells = PlaceCells( options, polygon)

    # trajectory simmulation
    trajectory_generator = TrajectoryGenerator( options, place_cells, polygon )

    # load model
    """ 

        load model

    """

    model = RNN( options, place_cells )

    print('\n')
    print('model: ')
    print( model )
    print('\n')

    checkpoint_file = precomputed_path + 'most_recent_model_cube_small_10_10_100_1000.pth'

    print(f'checkpoint file: {checkpoint_file}')

    checkpoint = torch.load( checkpoint_file, map_location=torch.device('cpu') )
    model.load_state_dict( checkpoint  )
    model = model.to( options.device )

    print('model loaded...')

    n_avg = 100
    res = 100
    Ng = options.Ng

    activations, rate_map, g, pos = compute_ratemaps(
        model=model, 
        trajectory_generator=trajectory_generator,
        polygon=polygon,
        options=options,
        res=res,
        n_avg=n_avg, 
        Ng=Ng
    )

    print(f'activations: {activations.shape}')
    print(f'rate map: {rate_map.shape}')
    print(f'g: {g.shape}')
    print(f'pos: {pos.shape}')

    rate_map_chunks = np.split( rate_map, 10 )

    for i, chunk in enumerate( rate_map_chunks ):
        
        np.save( f'rate_map_{i}.npy', chunk )