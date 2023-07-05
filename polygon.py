import numpy as np

def cube( side_length ):

    center = np.array([0, 0, 0])

    half_side = side_length / 2.0

    vertices = np.array(
        [
            [ -half_side, -half_side, -half_side ], 
            [ -half_side, -half_side, half_side ],
            [ -half_side,  half_side, -half_side ],
            [ -half_side,  half_side, half_side ],
            [ half_side, -half_side,  -half_side ],
            [ half_side, -half_side,  half_side ],
            [ half_side,  half_side,  -half_side ],
            [ half_side,  half_side,  half_side ]
        ]
    )

    boundaries = center + vertices

    return boundaries

def get_polygon( polygon ):

    if polygon == 'cube':

        return cube( 2.2 )