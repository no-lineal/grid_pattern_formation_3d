import torch

from polygon import get_polygon

from place_cells import PlaceCells
from trajectory_generator import TrajectoryGenerator

from model import RNN
from trainer import Trainer

import json

import os
import argparse

# viz
from matplotlib import pyplot as plt
import plotly.graph_objects as go

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

def plot_place_cells( place_cells, polygon, options ):

    # place cell centers
    us = place_cells.us

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter( us.cpu()[:,0], us.cpu()[:,1], us.cpu()[:,2], c='lightgrey', label='Place cell centers' )

    plt.savefig( options.save_path + options.model_name + 'place_cells.png' )

def plot_trajectory( place_cells, trajectory_generator, polygon, options ):

    # place cell centers
    us = place_cells.us

    # get test batch
    inputs, pos, pc_outputs = trajectory_generator.get_test_batch()
    pos = pos.cpu()

    data = []
    trace0 = go.Scatter3d(
        x = us.cpu()[:,0],
        y = us.cpu()[:,1],
        z = us.cpu()[:,2],
        mode = 'markers',
        marker = dict(size=4, color='blue', opacity=0.7),
        name = 'place cells', 
        opacity=0.5
    )
        
    data.append(trace0)

    # Create a trace for the trajectory
    for i in range( pos.cpu().shape[1] ):
        data.append(
            go.Scatter3d(
                x = pos.cpu()[:, i, 0],  # your x coordinates
                y = pos.cpu()[:, i, 1],
                z = pos.cpu()[:, i, 2],
                mode = 'lines',
                line = dict(color='red', width=2),
                name = 'trajectory ' + str(i), 
                opacity=0.5
            )
        )

    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.write_html( options.save_path + options.model_name + 'trajectory.html' )

if __name__ == '__main__':

    # test trajectory
    test_trajectory = False

    # where am i?
    PATH = os.getcwd() + '/'
    input_file = PATH + 'experiments/cube.json'

    print(f'PATH: {PATH}')
    print(f'experiment: {input_file}')

    # load JSON file
    with open( input_file ) as f:
        parameters = json.load( f )

    # generate options
    options = generate_options( parameters )

    # create save directory
    try:
        os.mkdir( options.save_path + options.model_name )
    except:
        pass

    # get polygon
    polygon = get_polygon( options.shape )

    # place cells
    place_cells = PlaceCells( options, polygon)

    # plot place
    plot_place_cells( place_cells, polygon, options )

    # trajectory simmulation
    trajectory_generator = TrajectoryGenerator( options, place_cells, polygon )

    # plot test trajectory
    if test_trajectory:

        plot_trajectory( place_cells, trajectory_generator, polygon, options )

    # model
    model = RNN( options, place_cells )
    model = model.to( options.device )

    # train
    trainer = Trainer( options, model, trajectory_generator, polygon, restore=False )
    trainer.train( n_epochs=options.n_epochs, n_steps=options.n_steps )