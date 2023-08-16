# -*- coding: utf-8 -*-
import torch

class RNN(torch.nn.Module):

    def __init__(self, options, place_cells):

        super(RNN, self).__init__()

        self.Ng = options.Ng # number of grid cells
        self.Np = options.Np # number of place cells

        self.sequence_length = options.sequence_length # number of steps in trajectory

        self.weight_decay = options.weight_decay # regularization
        self.place_cells = place_cells # place cell matrix, Npx2

        # Input weights

        # linear input layer
        self.encoder = torch.nn.Linear(
            in_features=self.Np, 
            out_features=self.Ng, 
            bias=False
        )
        
        # recurrent layer
        self.RNN = torch.nn.RNN(
            input_size=3, 
            hidden_size=self.Ng, 
            nonlinearity=options.activation, 
            bias=False
        )

        # linear read-out weights
        self.decoder = torch.nn.Linear(
            in_features=self.Ng, 
            out_features=self.Np, 
            bias=False
        )
        
        # output layer (probability distribution)
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        
        """

        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].

        """

        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g, _ = self.RNN(v, init_state)

        return g
    

    def predict(self, inputs):


        """

        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].

        """

        place_preds = self.decoder( self.g(inputs) )
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):

        """

        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.

        """

        eps = 1e-38

        y = pc_outputs

        preds = self.predict(inputs)
        yhat = self.softmax( preds )

        loss = -( y * torch.log(yhat + eps) ).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err