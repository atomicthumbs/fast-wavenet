from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

import pickle 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=5,
                     help='Number of hidden layers')
parser.add_argument('--learning_rate', type=float, default=0.001,
                     help='learning rate')
parser.add_argument('--stopping_loss', type=float, default=0.1,
		     help='loss at which training stops')
FLAGS, unparsed = parser.parse_known_args()


SAMPLE_RATE = 24000

inputs, targets = make_batch('assets/SMvocals.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 1

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction,
              num_layers=FLAGS.num_layers or 5,
	      learning_rate = FLAGS.learning_rate,
	      stopping_loss = FLAGS.stopping_loss)

tic = time()
model.train(inputs, targets)
toc = time()

print('Training took {} seconds.'.format(toc-tic))

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_, SAMPLE_RATE)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))

OUTPUT_PATH = 'generated_file'
pickle.dump(OUTPUT_PATH,predictions)
