![microWakeWord logo](etc/logo.png)

microWakeWord is an open-source wakeword library for detecting custom wake words on low power devices. It produces models that are suitable for using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). The models are suitable for real-world usage with low false accept and false reject rates.

**microWakeword is currently available as an early release. Training new models is intended for advanced users. Training a model that works well is still very difficult, as it typically requires experimentation with hyperparameters and sample generation settings. Please share any insights you find for training a good model!**

## Detection Process

We detect the wake word in two stages. Raw audio data is processed into 40 spectrogram features every 10 ms. The streaming inference model uses the newest slice of feature data as input and returns a probability that the wake word is said. If the model consistently predicts the wake word over multiple windows, then we predict that the wake word has been said.

The first stage processes the raw monochannel audio data at a sample rate of 16 kHz via the [micro_speech preprocessor](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). The preprocessor generates 40 features over 30 ms (the window duration) of audio data. The preprocessor generates these features every 10 ms (the stride duration), so the first 20 ms of audio data is part of the previous window. This process is similar to calculating a Mel spectrogram for the audio data, but it includes noise supression and automatic gain control. This makes it suitable for devices with limited processing power. See the linked TFLite Micro example for full details on how the audio is processed.

The streaming model performs inferences every 30 ms, where the initial convolution layer strides over three 10 ms slices of audio. The model is a neural network using [MixConv](https://arxiv.org/abs/1907.09595) mixed depthwise convolutions suitable for streaming. Streaming and training the model uses heavily modified open-sourced code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming) found in the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/pdf/2005.06720.pdf) by Rykabov, Kononenko, Subrahmanya, Visontai, and Laurenzo.

### Training Process
- We augment the spectrograms in several possible ways during training:
    - [SpecAugment](https://arxiv.org/pdf/1904.08779.pdf) masks time and frequency features
- The best weights are chosen as a two-step process:
    1. The top priority is minimizing a specific metric like the false accepts per hour on ambient background noise first.
    2. If the specified minimization target metric is met, then we maximize a different specified metric like accuracy.
- Validation and test sets are split into two portions:
    1. The ``validation`` and ``testing`` sets include the positive and negative generated samples.
    2. The ``validation_ambient`` and ``testing_ambient`` sets are all negative samples representing real-world background sounds; e.g., music, random household noises, and general speech/conversations.
- Generated spectrograms are stored as [Ragged Mmap](https://github.com/hristo-vrigazov/mmap.ninja/tree/master) folders for quick loading from the disk while training.
- Each feature set is configured with a ``sampling_weight`` and ``penalty_weight``. The ``sampling_weight`` parameter controls oversampling and ``penalty_weight`` controls the weight of incorrect predictions.
- Class weights are also adjustable with the ``positive_class_weight`` and ``negative_class_weight`` parameters. It is useful to increase the ``negative_class_weight`` to reduce the amount of false acceptances.
- We train the model in a non-streaming mode; i.e., it trains on the entire spectrogram. When finished, this is converted to a streaming model that updates on only the newest spectrogram features.
    - Not padding the convolutions ensures the non-streaming and streaming models have nearly identical prediction behaviors.
    - We estimate the false accepts per hour metric during training by splitting long-duration ambient clips into appropriate-sized spectrograms with a 100 ms stride to simulate the streaming model. This is not a perfect estimate of the streaming model's real-world false accepts per hour, but it is sufficient for determining the best weights.
- We should generate spectrogram features over a longer time period than needed for training the model. The preprocessor model applies PCAN and noise reduction, and generating features over a longer time period results in models that are better to generalize.
- We quantize the streaming models to increase performance on low-power devices. This has a small performance penalty that varies from model to model, but there is typically no reduction in accuracy.


## Model Training Process

We generate samples using [Piper sample generator](https://github.com/rhasspy/piper-sample-generator).

The generated samples are augmented before or during training to increase variability. There are pre-generated spectrogram features for various negative datasets available on [Hugging Face](https://huggingface.co/datasets/kahrendt/microwakeword).

Please see the ``basic_training_notebook.ipynb`` notebook to see how a model is trained. This notebook will produce a model, but it will most likely not be usable! Training a usable model requires a lot of experimentation, and that notebook is meant to serve only as a starting point for advanced users.

## Models

See https://github.com/esphome/micro-wake-word-models to download the currently available models.

## Acknowledgements

I am very thankful for many people's support to help improve this! Thank you, in particular, to the following individuals and organizations for providing feedback, collaboration, and developmental support:

  - [balloob](https://github.com/balloob)
  - [dscripka](https://github.com/dscripka)
  - [jesserockz](https://github.com/jesserockz)
  - [kbx81](https://github.com/kbx81)
  - [synesthesiam](https://github.com/synesthesiam)
  - [ESPHome](https://github.com/esphome)
  - [Nabu Casa](https://github.com/NabuCasa)
  - [Open Home Foundation](https://www.openhomefoundation.org/)