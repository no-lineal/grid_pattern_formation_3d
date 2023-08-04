import torch
import numpy as np
import scipy.interpolate as interpolate

class PlaceCells( object ):

    def __init__(self, options, polygon, us=None):

        self.load_path = options.load_path
        self.save_path = options.save_path

        self.Np = options.Np
        self.sigma = options.sigma # width of place cell center tuning curve (m)
        self.surround_scale = options.surround_scale
        self.periodic  = options.periodic
        self.DoG = options.DoG

        self.device = options.device

        # environment boundaries
        self.polygon = polygon

        self.softmax = torch.nn.Softmax( dim=-1 )

        # random seed
        np.random.seed( 0 )

        min_values = np.min(self.polygon, axis=0)
        max_values = np.max(self.polygon, axis=0)

        points = []
        while len(points) < self.Np:

            x = np.random.uniform(min_values[0], max_values[0])
            y = np.random.uniform(min_values[1], max_values[1])
            z = np.random.uniform(min_values[2], max_values[2])

            point = [x, y, z]

            if np.all(point >= min_values) and np.all(point <= max_values):

                points.append(point)

        self.us = torch.tensor(
            points
        )
                
        self.us = self.us.to( self.device )

    def get_activation(self, pos):
        

        '''

        Get place cell activations for a given position.

        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].

        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].

        '''

        d = torch.abs(pos[:, :, None, :] - self.us[None, None, ...]).float()

        if self.periodic:
                
            dx = d[:,:,:,0]
            dy = d[:,:,:,1]
            dz = d[:,:,:,2]

            dx = torch.minimum(dx, self.box_width - dx) 
            dy = torch.minimum(dy, self.box_height - dy)
            dz = torch.minimum(dz, self.box_depth - dz)

            d = torch.stack( [ dx, dy, dz] , axis=-1 )

        norm2 = (d**2).sum(-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on 
        # average and seems to speed up training.
        outputs = self.softmax( -norm2 / ( 2 * self.sigma**2 ) )

        if self.DoG:

            # Again, normalize with prefactor 
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= self.softmax(-norm2/(2*self.surround_scale*self.sigma**2))

            # Shift and scale outputs so that they lie in [0,1].
            min_output,_ = outputs.min(-1,keepdims=True)
            outputs += torch.abs(min_output)
            outputs /= outputs.sum(-1, keepdims=True)

        return outputs
    
    def get_nearest_cell_pos(self, activation, k=8):

        """

        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.

        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].

        """
        _, idxs = torch.topk(activation, k=k)
        pred_pos = self.us[idxs].mean(-2)
        
        return pred_pos

    def grid_pc( self, pc_outputs, res=32 ):

        """
        
        Grid place cell outputs.interpolate place cell outputs onto a grid
        
        """

        min_x, min_y, min_z = self.polygon.min(0)
        max_x, max_y, max_z = self.polygon.max(0)

        width = max_x - min_x
        height = max_y - min_y
        depth = max_z - min_z

        coors_x = torch.linspace(min_x, max_x, res)
        coors_y = torch.linspace(min_y, max_y, res)
        coors_z = torch.linspace(min_z, max_z, res)

        grid_x, grid_y, grid_z = torch.meshgrid(coors_x, coors_y, coors_z)
        grid = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        # Convert to numpy
        pc_outputs = pc_outputs.reshape(-1, self.Np)

        T = pc_outputs.shape[0]
        pc = np.zeros((T, res, res, res))

        for i in range( len(pc_outputs) ):

            gridval = interpolate.griddata( self.us.cpu(), pc_outputs[i].cpu(), grid )
            pc[ i ] = gridval.reshape( [ res, res, res ] )

        return pc