# microWakeWord

microWakeWord is an open-source wakeword library for detecting custom wake words on low power devices. It produces models that are suitable for using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). The models are suitable for real-world usage with low false accept and false reject rates.

**microWakeword is currently available as a very early release. microWakeWord can generate features and train models. It does not include sample generation or audio augmentations. The training process produces usable models if you manually fine-tune penalty weights.**

## Detection Process

We detect the wake word in two stages. Raw audio data is processed into 40 features every 20 ms. These features construct a spectrogram. The streaming inference model uses the newest slice of feature data as input and returns a probability that the wake word is said. If the model consistently predicts the wake word over multiple windows, then we predict that the wake word has been said.

The first stage processes the raw monochannel audio data at a sample rate of 16 kHz via the [micro_speech preprocessor](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). The preprocessor generates 40 features over 30 ms (the window duration) of audio data. The preprocessor generates these features every 20 ms (the stride duration), so the first 10 ms of audio data is part of the previous window. This process is similar to calculating a Mel spectrogram for the audio data, but it is lightweight for devices with limited processing power. See the linked TFLite Micro example for full details on how the audio is processed.

The streaming model performs inferences every 20 ms on the newest audio stride. The model is based on an [inception neural network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202?gi=6bc760f44aef) converted for streaming. Streaming and training the model uses heavily modified open-sourced code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming) found in the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/pdf/2005.06720.pdf) by Rykabov, Kononenko, Subrahmanya, Visontai, and Laurenzo.

## Model and Training Design Notes

### Inception Based Model
- We apply [SubSpectral Normalization](https://arxiv.org/abs/2103.13620) after the initial convolution layer
- Temporal dilatations for later convolutions greatly improve accuracy; these operations are currently not optimized in the TFLite Micro for Espressif's chip, so by default are not configured
- The model doesn't use a Global Average Pooling layer, but rather a larger Fully Connected layer. This improves accuracy, and it is much faster on ESP32 devices.
- Some wake word phrases may not need as large of a model. Adjusting ``cnn1_filters``, ``cnn2_filters1``, and ``cnn2_filters2`` can increase or decrease the model size and latency.
- All convolutions have no padding. The training process ensures the last layer has features representing exactly ``clip_duration_ms``.

### Training Process
- We augment the spectrograms in several possible ways during training:
    - [SpecAugment](https://arxiv.org/pdf/1904.08779.pdf) masks time and frequency features
    - [MixUp](https://openreview.net/forum?id=r1Ddp1-Rb) averages two spectrograms and their labels
    - [FreqMix](https://arxiv.org/pdf/2204.11479.pdf) combines two spectrograms and their labels using a low-pass and high-pass filter.
- The best weights are chosen as a two-step process:
    1. The top priority is minimizing a specific metric like the false accepts per hour on ambient background noise first.
    2. If the specified minimization target metric is met, then we maximize a different specified metric like accuracy.
- Validation and test sets are split into two portions:
    1. The ``validation`` and ``testing`` sets include the positive and negative generated samples.
    2. The ``validation_ambient`` and ``testing_ambient`` sets are all negative samples representing real-world background sounds; e.g., music, random household noises, and general speech/conversations.
- Generated spectrograms are stored as [Ragged Mmap](https://github.com/hristo-vrigazov/mmap.ninja/tree/master) folders for quick loading from the disk while training.
- Each feature set is configured with a ``sampling_weight`` and ``penalty_weight``. The ``sampling_weight`` parameter controls oversampling and ``penalty_weight`` controls the weight of incorrect predictions.
- Class weights are also adjustable with the ``positive_class_weight`` and ``negative_class_weight`` parameters. It is useful to increase the ``negative_class_weight`` near the end of the training process to reduce the amount of false accepts.
- We train the model in a non-streaming mode; i.e., it trains on the entire spectrogram. When finished, this is converted to a streaming model that updates every 20 ms.
    - Not padding the convolutions ensures the non-streaming and streaming models have nearly identical prediction behaviors.
    - We estimate the false accepts per hour metric during training by splitting long-duration ambient clips into appropriate-sized spectrograms with a 100 ms stride to simulate the streaming model. This is not a perfect estimate of the streaming model's real-world false accepts per hour, but it is sufficient for determining the best weights.
- We should generate spectrogram features over a longer time period than needed for training the model. The preprocessor model applies PCAN and noise reduction, and generating features over a longer time period results in models that are better to generalize. _This is not currently automatically implemented in microWakeWord._
- We quantize the streaming models to increase performance on low-power devices. This has a small performance penalty that varies from model to model, but it typically lowers accuracy on the test dataset by around 0.05%.

## Benchmarks

Benchmarking and comparing wake word models is challenging. It is hard to account for all the different operating environments. [Picovoice](https://github.com/Picovoice/wake-word-benchmark) has provided one benchmark for at least one point of comparison.

The following graph depicts the false-accept/false-reject rate for the "Hey Jarvis" model. Note that the test clips used in the benchmark are created with Piper sample generator, not real voice samples.
![FPR/FRR curve for "hey jarvis" pre-trained model](benchmarks/hey_jarvis_roc_curve.png)


The default parameters (probablity cutoff of 0.5 and average window size of 10) has a false rejection rate of 0.67% and 0.081 false accepts per hour with the Picovoice benchmark dataset.

For a more rigorous false acceptance metric, we tested the "Hey Jarvis" on the [Dinner Party Corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus) dataset. The component's default configuration values result in a 0.375 false accepts per hour.

## Model Training Process

We generate positive and negative samples using [openWakeWord](https://github.com/dscripka/openWakeWord), which relies on [Piper sample generator](https://github.com/rhasspy/piper-sample-generator). We also use openWakeWord's data tools to augment the positive and negative samples. Additional data sources are used for negative data. Currently, microWakeWord does support these steps directly.

Audio samples are converted to features are stored as Ragged Mmaps. Currently, only converting wav audio files are supported and no direct audio augmentations are applied.

```python
from microwakeword.feature_generation import generate_features_for_folder

generate_features_for_folder(path_to_audio='audio_samples/training', features_output_dir='audio_features/training', set_name='audio_samples')
```

Training configuration options are stored in a yaml file.

```python
# Save a yaml config that controls the training process
import yaml
import os

config = {}

config['train_dir'] = 'trained_models/alexa'

# Each feature_dir should have at least one of the following folders with this structure:
#  training/
#    ragged_mmap_folders_ending_in_mmap
#  testing/
#    ragged_mmap_folders_ending_in_mmap
#  testing_ambient/
#    ragged_mmap_folders_ending_in_mmap
#  validation/
#    ragged_mmap_folders_ending_in_mmap
#  validation_ambient/
#    ragged_mmap_folders_ending_in_mmap
#
#  sampling_weight: Weight for choosing a spectrogram from this set in the batch
#  penalty_weight: Penalizing weight for incorrect predictions from this set
#  truth: Boolean whether this set has positive samples or negative samples
#  truncation_strategy = If spectrograms in the set are longer than necessary for training, how are they truncated
#       - random: choose a random portion of the entire spectrogram - useful for long negative samples
#       - truncate_start: remove the start of the spectrogram
#       - truncate_end: remove the end of the spectrogram
#       - split: Split the longer spectrogram into separate spectrograms offset by 100 ms. Only for ambient sets

config['features'] = [
        {
            'features_dir': '/Volumes/MachineLearning/training_data/alexa_4990ms_spectrogram/generated_positive',
            'sampling_weight': 0.25,
            'penalty_weight': 1,
            'truth': True,
            'truncation_strategy': 'truncate_start'
        },
        {
            'features_dir': '/Volumes/MachineLearning/training_data/alexa_4990ms_spectrogram/generated_negative',
            'sampling_weight': 0.25,
            'penalty_weight': 1,
            'truth': False,
            'truncation_strategy': 'truncate_start'
        },
        {
            'features_dir': '/Volumes/MachineLearning/training_data/english_speech_background_1970ms',
            'sampling_weight': 0.2,
            'penalty_weight': 3,
            'truth': False,
            'truncation_strategy': 'random'
        },
        {
            'features_dir': '/Volumes/MachineLearning/training_data/cv_corpus_background',
            'sampling_weight': 0.10,
            'penalty_weight': 2,
            'truth': False,
            'truncation_strategy': 'random'
        },
        {
            'features_dir': '/Volumes/MachineLearning/training_data/no_speech_background_1970ms',
            'sampling_weight': 0.2,
            'penalty_weight': 3,
            'truth': False,
            'truncation_strategy': 'random'
        },
        {
            'features_dir': '/Volumes/MachineLearning/training_data/ambient_background',
            'sampling_weight': 0.2,
            'penalty_weight': 2,
            'truth': False,
            'truncation_strategy': 'split'
        },
    ]

# Number of training steps in each iteration - various other settings are configured as lists that corresponds to different steps
config['training_steps'] = [20000, 20000, 20000]        

# Penalizing weight for incorrect class predictions - lists that correspond to training steps
config["positive_class_weight"] = [1]               
config["negative_class_weight"] = [1]
config['learning_rates'] = [0.001, 0.0005, 0.00025]     # Learning rates for Adam optimizer - list that corresponds to training steps
config['batch_size'] = 100

config['mix_up_augmentation_prob'] =  [0]       # Probability of applying MixUp augmentation - list that corresponds to training steps
config['freq_mix_augmentation_prob'] = [0]      # Probability of applying FreqMix augmentation - list that corresponds to training steps
config['time_mask_max_size'] = [5]              # SpecAugment - list that corresponds to training steps
config['time_mask_count'] = [2]                 # SpecAugment - list that corresponds to training steps
config['freq_mask_max_size'] = [5]              # SpecAugment - list that corresponds to training steps
config['freq_mask_count'] = [2]                 # SpecAugment - list that corresponds to training steps
config['eval_step_interval'] = 500              # Test the validation sets after every this many steps

config['clip_duration_ms'] = 1490   # Maximum length of wake word that the streaming model will accept
config['window_stride_ms'] = 20     # Fixed setting for default feature generator
config['window_size_ms'] = 30       # Fixed setting for default feature generator
config['sample_rate'] = 16000       # Fixed setting for default feature generator

# The best model weights are chosen first by minimizing the specified minimization metric below the specified target_minimization
# Once the target has been met, it chooses the maximum of the maximization metric. Set 'minimization_metric' to None to only maximize
# Available metrics:
#   - "loss" - cross entropy error on validation set
#   - "accuracy" - accuracy of validation set
#   - "recall" - recall of validation set
#   - "precision" - precision of validation set
#   - "false_positive_rate" - false positive rate of validation set
#   - "false_negative_rate" - false negative rate of validation set
#   - "ambient_false_positives" - count of false positives from the split validation_ambient set
#   - "ambient_false_positives_per_hour" - estimated number of false positives per hour on the split validation_ambient set
config['minimization_metric'] = 'ambient_false_positives_per_hour'  # Set to N
config['target_minimization'] = 0.5
config['maximization_metric'] = 'accuracy'

with open(os.path.join('training_parameters.yaml'), 'w') as file:
    documents = yaml.dump(config, file)
```

The model's hyperparameters are specified when calling the training script.

```python
!python -m microwakeword.model_train_eval \
--training_config='training_parameters.yaml' \
--train 1 \
--restore_checkpoint 1 \
--test_tf_nonstreaming 0 \
--test_tflite_nonstreaming 0 \
--test_tflite_streaming 0 \
--test_tflite_streaming_quantized 1 \
inception \
--cnn1_filters '32' \
--cnn1_kernel_sizes '5' \
--cnn1_subspectral_groups '4' \
--cnn2_filters1 '24,24,24' \
--cnn2_filters2 '32,64,96' \
--cnn2_kernel_sizes '3,5,5' \
--cnn2_subspectral_groups '1,1,1' \
--cnn2_dilation '1,1,1' \
--dropout 0.8 
```

## Acknowledgements

I am very thankful for many people's support to help improve this! Thank you, in particular, to the following individuals and organizations for providing feedback, collaboration, and developmental support:

  - [balloob](https://github.com/balloob)
  - [dscripka](https://github.com/dscripka)
  - [jesserockz](https://github.com/jesserockz)
  - [kbx81](https://github.com/kbx81)
  - [synesthesiam](https://github.com/synesthesiam)
  - [ESPHome](https://github.com/esphome)
  - [Nabu Casa](https://github.com/NabuCasa)