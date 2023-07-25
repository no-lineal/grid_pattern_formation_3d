# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps
import os

import json

torch.autograd.set_detect_anomaly(True)

class Trainer(object):

    def __init__(self, options, model, trajectory_generator, polygon, restore=True):

        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        self.polygon = polygon

        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        ### 
        if torch.cuda.device_count() > 1:

            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = torch.nn.DataParallel( self.model )

        self.model = self.model.to( options.device )
        ###

        # Set up checkpoints

        self.ckpt_dir = os.path.join(options.save_path, options.model_name)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')

        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):

            self.model.load_state_dict(torch.load(ckpt_path))
            print("Restored trained model from {}".format(ckpt_path))

        else:

            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)

            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def train_step(self, inputs, pc_outputs, pos):

        """

        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.

        """

        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        # gradient exploding
        #torch.nn.utils.clip_grad_norm_( self.model.parameters(), 0.5 )
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):

        """
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        """

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)

        loss_dict = {}
        for epoch_idx in range(n_epochs):
            
            loss_dict[ epoch_idx ] = {}
            for step_idx in range(n_steps):

                inputs, pc_outputs, pos = next(gen)

                loss, err = self.train_step(inputs, pc_outputs, pos)

                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                print('Epoch: {}/{}. Step {}/{}. Loss: {}. Err: {}cm'.format(
                    epoch_idx + 1, n_epochs, step_idx + 1, n_steps,
                    np.round(loss, 2), np.round(100 * err, 2)))
                
                loss_dict[ epoch_idx ][ step_idx ] = { 'loss': loss, 'err': err }

            if save:
                # Save checkpoint
                ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                 'most_recent_model.pth'))

                # Save a picture of rate maps
                save_ratemaps(model=self.model, trajectory_generator=self.trajectory_generator, polygon=self.polygon, options=self.options, step=epoch_idx)

        # Save loss and error
        with open( os.path.join( self.ckpt_dir, 'loss.json' ), 'w' ) as f:
            json.dump( loss_dict, f )