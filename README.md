# microWakeWord

microWakeWord is an open-source wakeword library for detecting custom wake words on low power devices. It produces models that are suitable for using [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers). The models are suitable for real-world usage with low false accept and false reject rates.

## Detection Process

We detect the wake word in two stages. Raw audio data is processed into 40 features every 20 ms. Several of these features construct a spectrogram. The streaming inference model uses the newest slice of feature data as input and returns a probability that the wake word is said. If the model consistently predicts the wake word over multiple windows, then we predict that the wake word has been said.

The first stage processes the raw monochannel audio data at a sample rate of 16 kHz via the [micro_speech preprocessor](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech). The preprocessor generates 40 features over 30 ms (the window duration) of audio data. The preprocessor generates these features every 20 ms (the stride duration), so the first 10 ms of audio data is part of the previous window. This process is similar to calculating a Mel spectrogram for the audio data, but it is lightweight for devices with limited processing power. See the linked TFLite Micro example for full details on how the audio is processed.

The streaming model performs inferences every 20 ms on the newest audio stride. The model is an [inceptional neural network](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202?gi=6bc760f44aef) converted for streaming. Streaming and training the model uses modified open-sourced code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming) found in the paper [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/pdf/2005.06720.pdf) by Rykabov, Kononenko, Subrahmanya, Visontai, and Laurenzo.

## Model Training Process

We generate positive and negative samples using [openWakeWord](https://github.com/dscripka/openWakeWord), which relies on [Piper sample generator](https://github.com/rhasspy/piper-sample-generator). We also use openWakeWord's data tools to augment the positive and negative samples. Then, we train the two models using code from [Google Research](https://github.com/google-research/google-research/tree/master/kws_streaming). The streaming model is an inception neural network converted for streaming.

The models' parameters are set via

```
TRAIN_DIR = 'trained_models/inception'
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1490
FEATURE_BIN_COUNT = 40
BACKGROUND_FREQUENCY = 0
BACKGROUND_VOLUME_RANGE = 0
TIME_SHIFT_MS = 0.0
WINDOW_STRIDE = 20
WINDOW_SIZE_MS = 30
PREPROCESS = 'none'
WANTED_WORDS = "wakeword,unknown"
DATASET_DIR =  'data_training'
```

We train the streaming model using the follow (note it requires several modification's to the default code that are included in the ``kws_streaming`` folder.)

```
!python -m kws_streaming.train.model_train_eval \
--data_url='' \
--data_dir={DATASET_DIR} \
--train_dir={TRAIN_DIR} \
--split_data 0 \
--mel_upper_edge_hertz 7500.0 \
--mel_lower_edge_hertz 125.0 \
--how_many_training_steps 10000 \
--learning_rate 0.001 \
--window_size_ms={WINDOW_SIZE_MS} \
--window_stride_ms={WINDOW_STRIDE} \
--clip_duration_ms={CLIP_DURATION_MS} \
--eval_step_interval=500 \
--mel_num_bins={FEATURE_BIN_COUNT} \
--dct_num_features={FEATURE_BIN_COUNT} \
--preprocess={PREPROCESS} \
--alsologtostderr \
--train 1 \
--wanted_words={WANTED_WORDS} \
--pick_deterministically 0 \
--return_softmax 1 \
--restore_checkpoint 0 \
inception \
--cnn1_filters '32' \
--cnn1_kernel_sizes '5' \
--cnn1_strides '1' \
--cnn2_filters1 '16,16,16' \
--cnn2_filters2 '32,64,70' \
--cnn2_kernel_sizes '3,5,5' \
--cnn2_strides '1,1,1' \
--dropout 0.0 \
--bn_scale 0
```