import torch
import numpy as np

class TrajectoryGenerator( object ):

    def __init__(self, options, place_cells, polygon):

        self.device = options.device
        self.sequence_length = options.sequence_length
        self.periodic = options.periodic
        self.batch_size = options.batch_size

        self.place_cells = place_cells
        self.polygon = polygon

        self.border_region = 0.03 # meters

    def generate_trajectory( self, batch_size ):
        
        """
        
            generate a random walk trajectory
            
        """
        
        samples = self.sequence_length # steps in trajectory
        
        dt = 0.02 # time step increment (seconds)
        sigma = 5.76 * 2 # standard deviation of rotation velocity (rads / second)
        b = 0.13 * 2 * np.pi # forward velocity rayleigh distribution scale (m/sec)
        mu = 0.0 # turn angle bias

        min = np.min( self.polygon, axis=0 )
        max = np.max( self.polygon, axis=0 )

        # starting points
        start_points = []
        while len(start_points) < batch_size:
            
            x = np.random.uniform(min[0], max[0])
            y = np.random.uniform(min[1], max[1])
            z = np.random.uniform(min[2], max[2])
            
            point = np.array( [ x, y, z ] )
            
            if np.all(np.logical_and( self.polygon.min(axis=0) <= point, point <= self.polygon.max(axis=0))):
                
                start_points.append( point )

        # empty space
        position = np.zeros( [ batch_size, samples + 2, self.polygon.shape[1] ] ) # batch, steps, (x, y, z)
        head_direction = np.zeros( [ batch_size, samples + 2, self.polygon.shape[1] ] ) # batch, steps, (roll, pitch, yaw)
        velocity = np.zeros( [ batch_size, samples + 2 ] ) # batch, steps

        # initialize position
        position[:, 0, 0] = np.array( [ point[0] for point in start_points ] )
        position[:, 0, 1] = np.array( [ point[1] for point in start_points ] )
        position[:, 0, 2] = np.array( [ point[2] for point in start_points ] )

        # initialize head direction
        #head_direction[:, 0, 0] = np.random.uniform( - np.pi, np.pi, batch_size) # roll
        #head_direction[:, 0, 1] = np.random.uniform( - np.pi / 6, np.pi / 6, batch_size) # pitch
        #head_direction[:, 0, 2] = np.random.uniform( - np.pi, np.pi, batch_size) # yaw

        # limited range of movement
        head_direction[:, 0, 0] = np.random.uniform( - np.pi, np.pi, batch_size) # roll
        head_direction[:, 0, 1] = np.random.uniform( - np.pi / 2, np.pi / 2, batch_size) # pitch
        head_direction[:, 0, 2] = np.random.uniform( - np.pi, np.pi, batch_size) # yaw

        # initial velocity
        random_vel = np.random.rayleigh( b, [ batch_size, samples + 1 ] )

        # path integration
        for t in range( samples + 1 ):

            # random velocity
            v = random_vel[:, t]

            # take a step
            velocity[:, t] = v * dt 

            # update position
            position[ :, t + 1, 0 ] = position[ :, t, 0 ] + velocity[:, t] * ( np.cos( head_direction[ :, t, 1 ] * np.cos( head_direction[ :, t, 2 ] ) ) ) # update roll
            position[ :, t + 1, 1 ] = position[ :, t, 1 ] + velocity[:, t] * ( np.cos( head_direction[ :, t, 1 ] * np.sin( head_direction[ :, t, 2 ] ) ) ) # update pitch
            position[ :, t + 1, 2 ] = position[ :, t, 2 ] + velocity[:, t] * ( np.sin( head_direction[ :, t, 1 ] ) ) # update yaw

            # update head direction
            head_direction[:, t + 1, 0] = head_direction[:, t, 0] + dt * 1
            head_direction[:, t + 1, 1] = head_direction[:, t, 1] + dt * 3
            head_direction[:, t + 1, 2] = head_direction[:, t, 2] + dt * 3

        trajectory = {}

        # input variables
        trajectory['init_roll'] = head_direction[:, 1, 0, None]
        trajectory['init_pitch'] = head_direction[:, 1, 1, None]
        trajectory['init_yaw'] = head_direction[:, 1, 2, None]
        trajectory['init_x'] = position[:, 1, 0, None]
        trajectory['init_y'] = position[:, 1, 1, None]
        trajectory['init_z'] = position[:, 1, 2, None]
        trajectory['ego_v'] = velocity[:, 1:-1 ]

        # target variables
        trajectory['target_roll'] = head_direction[:, 2:, 0]
        trajectory['target_pitch'] = head_direction[:, 2:, 1]
        trajectory['target_yaw'] = head_direction[:, 2:, 2]
        trajectory['target_x'] = position[:, 2:, 0]
        trajectory['target_y'] = position[:, 2:, 1]
        trajectory['target_z'] = position[:, 2:, 2]
    
        return trajectory

    def get_test_batch( self, batch_size=None ):

        """

        test generator, return a batch of sample trajectories
        
        """

        if not batch_size:

            batch_size = self.batch_size

        trajectory = self.generate_trajectory( batch_size )

        # velocity
        v = np.stack(
            [
                trajectory['ego_v'] * np.cos(trajectory['target_yaw']) * np.cos(trajectory['target_pitch']), 
                trajectory['ego_v'] * np.sin(trajectory['target_yaw']) * np.cos(trajectory['target_pitch']), 
                trajectory['ego_v'] * np.sin(trajectory['target_pitch'])
            ], 
            axis=-1
        )

        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

        # position
        pos = np.stack( [ trajectory['target_x'], trajectory['target_y'], trajectory['target_z'] ], axis=-1 )
        pos = torch.tensor( pos, dtype=torch.float32 ).transpose(0, 1)
        pos = pos.to( self.device )

        # activation
        place_outputs = self.place_cells.get_activation(pos)

        # initial position
        init_pos = np.stack( [ trajectory['init_x'], trajectory['init_y'], trajectory['init_z'] ] , axis=-1 )
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to( self.device )

        # initial activation
        init_actv = self.place_cells.get_activation(init_pos).squeeze()

        v = v.to( self.device )
        inputs = (v, init_actv)

        return (inputs, pos, place_outputs)
    
    def get_generator( self, batch_size=None ):

        """
        
            return a generator that yields batches of trajectories
        
        """

        if not batch_size:

            batch_size = self.batch_size

        n = 0
        while True:

            trajectory = self.generate_trajectory( batch_size )

            # velocity
            v = np.stack(
                [
                    trajectory['ego_v'] * np.cos(trajectory['target_yaw']) * np.cos(trajectory['target_pitch']), 
                    trajectory['ego_v'] * np.sin(trajectory['target_yaw']) * np.cos(trajectory['target_pitch']), 
                    trajectory['ego_v'] * np.sin(trajectory['target_pitch'])
                ], 
                axis=-1
            )

            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)

            # position
            pos = np.stack( [ trajectory['target_x'], trajectory['target_y'], trajectory['target_z'] ], axis=-1 )
            pos = torch.tensor( pos, dtype=torch.float32 ).transpose(0, 1)
            pos = pos.to( self.device )

            # activation
            place_outputs = self.place_cells.get_activation(pos)

            # initial position
            init_pos = np.stack( [ trajectory['init_x'], trajectory['init_y'], trajectory['init_z'] ] , axis=-1 )
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to( self.device )

            # initial activation
            init_actv = self.place_cells.get_activation(init_pos).squeeze()

            v = v.to( self.device )
            inputs = (v, init_actv)

            n += 1

            yield (inputs, place_outputs, pos)