import glob
import json
import os
import random

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflowjs as tfjs
import tqdm

print(tf.__version__)
print(tfjs.__version__)

# Inspired from https://www.tensorflow.org/tutorials/audio/simple_audio & https://colab.research.google.com/github/tensorflow/tfjs-models/blob/master/speech-commands/training/browser-fft/training_custom_audio_model_in_python.ipynb

preproc_model_path = 'tfjs-sc-model/sc_preproc_model'
preproc_model = tf.keras.models.load_model(preproc_model_path)
print(preproc_model.summary())
print(preproc_model.input_shape)

# Only 1s
TARGET_SAMPLE_TIME = 1.0
# Target sampling rate. It is required by the audio preprocessing model.
TARGET_SAMPLE_RATE = 44100
# The specific audio tensor length expected by the preprocessing model.
EXPECTED_WAVEFORM_LEN = preproc_model.input_shape[-1]

# Where the Speech Commands v0.02 dataset has been downloaded.
DATA_ROOT = "final-data"

WORDS = ("_background_noise_", "Euuh", "Yolo", "Next", "Back")

def resample_wavs(dir_path, target_sample_rate=44100):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  resampled_suffix = "_%shz.wav" % target_sample_rate
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if wav_path.endswith(resampled_suffix) or 'split' not in wav_path:
      continue
    xs, sample_rate = librosa.load(wav_path, None)
    #xs = xs.astype(np.float32)
    xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE)#.astype(np.int16)
    resampled_path = os.path.splitext(wav_path)[0] + resampled_suffix
    sf.write(resampled_path, xs, target_sample_rate)


def add_noise(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if 'noise' in wav_path:
      continue
    if 'data_aug' in wav_path or 'hz.wav' in wav_path:
      continue
    wav, rate = librosa.load(wav_path, None)
    wav_n = wav + 0.009 * np.random.normal(0, 1, len(wav))
    sf.write(wav_path + '-data_aug_noise.wav', wav_n, rate)


def timeshift(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if 'timeshift' in wav_path:
      continue
    if 'data_aug' in wav_path or 'hz.wav' in wav_path:
      continue
    wav, rate = librosa.load(wav_path, None)
    wav_n = np.roll(wav, int(rate / 10))
    sf.write(wav_path + '-data_aug-timeshift.wav', wav_n, rate)


def split_to_time(dir_path):
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  for wav in wav_paths:
    if 'split' in wav:
      continue
    data, rate = librosa.load(wav, None)
    batches = int(len(data) / (TARGET_SAMPLE_TIME * rate))
    for i in range(batches):
      sf.write(wav + '-split' + '-' + str(i) + '.wav', data[i * rate: (i+1) * rate], rate)


@tf.function
def read_wav(filepath):
  file_contents = tf.io.read_file(filepath)
  audio = tf.audio.decode_wav(
    file_contents,
    desired_channels=1,
    desired_samples=TARGET_SAMPLE_RATE).audio
  return tf.expand_dims(tf.squeeze(audio, axis=-1), 0)

@tf.function
def filter_by_waveform_length(waveform, label):
  return tf.size(waveform) > EXPECTED_WAVEFORM_LEN


@tf.function
def crop_and_convert_to_spectrogram(waveform, label):
  cropped = tf.slice(waveform, begin=[0, 0], size=[1, EXPECTED_WAVEFORM_LEN])
  return tf.squeeze(preproc_model(cropped), axis=0), label

@tf.function
def spectrogram_elements_finite(spectrogram, label):
  return tf.math.reduce_all(tf.math.is_finite(spectrogram))


def get_dataset(input_wav_paths, labels):
  ds = tf.data.Dataset.from_tensor_slices(input_wav_paths)
  # Read audio waveform from the .wav files.
  ds = ds.map(read_wav)
  ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))
  # Keep only the waveforms longer than `EXPECTED_WAVEFORM_LEN`.
  ds = ds.filter(filter_by_waveform_length)
  # Crop the waveforms to `EXPECTED_WAVEFORM_LEN` and convert them to
  # spectrograms using the preprocessing layer.
  ds = ds.map(crop_and_convert_to_spectrogram)
  # Discard examples that contain infinite or NaN elements.
  ds = ds.filter(spectrogram_elements_finite)
  return ds

# Resample data
for word in WORDS:
  word_dir = os.path.join(DATA_ROOT, word)
  assert os.path.isdir(word_dir)
  # data augmentation
  print('data augmentation for %s' % word)
  split_to_time(word_dir)
  add_noise(word_dir)
  timeshift(word_dir)
  resample_wavs(word_dir, target_sample_rate=TARGET_SAMPLE_RATE)

input_wav_paths_and_labels = []
for i, word in enumerate(WORDS):
  wav_paths = glob.glob(os.path.join(DATA_ROOT, word, "*_%shz.wav" % TARGET_SAMPLE_RATE))
  print("Found %d examples for class %s" % (len(wav_paths), word))
  labels = [i] * len(wav_paths)
  input_wav_paths_and_labels.extend(zip(wav_paths, labels))
random.shuffle(input_wav_paths_and_labels)

input_wav_paths, labels = ([t[0] for t in input_wav_paths_and_labels],
                           [t[1] for t in input_wav_paths_and_labels])
dataset = get_dataset(input_wav_paths, labels)

# The amount of data we have is relatively small. It fits into typical host RAM
# or GPU memory. For better training performance, we preload the data and
# put it into numpy arrays:
# - xs: The audio features (normalized spectrograms).
# - ys: The labels (class indices).
print(
    "Loading dataset and converting data to numpy arrays. "
    "This may take a few minutes...")
xs_and_ys = list(dataset)
xs = np.stack([item[0] for item in xs_and_ys])
ys = np.stack([item[1] for item in xs_and_ys])

tfjs_model_json_path = 'tfjs-sc-model/model.json'

# Load the Speech Commands model. Weights are loaded along with the topology,
# since we train the model from scratch. Instead, we will perform transfer
# learning based on the model.
orig_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=True)

# Remove the top Dense layer and add a new Dense layer of which the output
# size fits the number of sound classes we care about.
model = tf.keras.Sequential(name="TransferLearnedModel")
for layer in orig_model.layers[:-1]:
  model.add(layer)
model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax"))

# Freeze all but the last layer of the model. The last layer will be fine-tuned
# during transfer learning.
for layer in model.layers[:-1]:
  layer.trainable = False

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
print(model.summary())

# Train the model.
model.fit(xs, ys, batch_size=256, validation_split=0.3, shuffle=True, epochs=60)

# Convert the model to TensorFlow.js Layers model format.

tfjs_model_dir = "tfjs-model"
tfjs.converters.save_keras_model(model, tfjs_model_dir)

# Create the metadata.json file.
metadata = {"words": ["_background_noise_"] + list(WORDS[1:]), "frameSize": model.input_shape[-2]}
with open(os.path.join(tfjs_model_dir, "metadata.json"), "w") as f:
  json.dump(metadata, f)
