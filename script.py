from time import time

from wavenet.utils import make_batch
from wavenet.models import Model, Generator

import librosa

SAMPLE_RATE = 24000

inputs, targets = make_batch('assets/SMvocals.wav')
num_time_samples = inputs.shape[1]
num_channels = 1
gpu_fraction = 0

model = Model(num_time_samples=num_time_samples,
              num_channels=num_channels,
              gpu_fraction=gpu_fraction)


tic = time()
model.train(inputs, targets)
toc = time()

print('Training took {} seconds.'.format(toc-tic))

generator = Generator(model)

# Get first sample of input
input_ = inputs[:, 0:1, 0]

tic = time()
predictions = generator.run(input_, 32000)
toc = time()
print('Generating took {} seconds.'.format(toc-tic))

OUTPUT_PATH = 'generated.wav'
librosa.output.write_wav(OUTPUT_PATH,predictions,SAMPLE_RATE)
