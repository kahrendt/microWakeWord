# Data Sources for Training Wake Words

## Generated Samples

[Piper sample generator](https://github.com/rhasspy/piper-sample-generator) uses text-to-speech to generate many wake word samples. We also generate adversarial phrase samples created using [openWakeWord](https://github.com/dscripka/openWakeWord).

## Augmentation Sources

We apply several augments to the generated samples. We use the following sources for background audio samples:

- [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475) - (Various Creative Commons Licenses.)
- [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840) - (Creative Commons Attribution 4.0 International License.)
- [WHAM!: Extending Speech Separation to Noisy Environments](https://arxiv.org/abs/1907.01160) - (Creative Commons Attribution-NonCommercial 4.0 International License.)

We reverberate the samples with room impulse responses from [BIRD: Big Impulse Response Dataset](https://arxiv.org/abs/2010.09930).

## Ambient Noises for Negative Samples

We use a variety of sources of ambient background noises as negative samples during training.

### Ambient Speech

- [Voices Obscured in Complex Environmental Settings (VOICES) corpus](https://arxiv.org/abs/1804.05053) - (Creative Commons Attribution 4.0 License.)
- [Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670) - (Creative Commons License.)

### Ambient Background

- [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475)
- [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840) - reverberated with room impulse responses
- [WHAM!: Extending Speech Separation to Noisy Environments](https://arxiv.org/abs/1907.01160)

## Validation and Test Sets

We generate positive and negative samples solely for validation and testing. We augment these samples in the same way as the training data. We split the FSDK50K, FMA, and WHAM! datasets 90-10 into training and testing sets (they are not in the validation set). We estimate the false accepts per hour during training with the VOiCES validation set and [DiPCo - Dinner Party Corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus) (Community Data License Agreement – Permissive Version 1.0 License.) We test the false accepts per hour in streaming mode after training with the DiPCo set.
