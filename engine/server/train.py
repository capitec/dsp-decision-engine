#!/usr/bin/env python

from __future__ import print_function

import os
import sys


# These are the paths to where SageMaker mounts interesting things in your container.
prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
channel_name='training'
training_path = os.path.join(input_path, channel_name)

# The function to execute the training.
def train():
    raise NotImplementedError("Currently no way to train a decisioning flow.")

if __name__ == '__main__':
    train()
    sys.exit(0)
