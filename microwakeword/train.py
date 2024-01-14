import os
import pprint

from absl import logging
from collections import deque

import numpy as np
import tensorflow as tf

import microwakeword.data as input_data
import microwakeword.inception as inception
import microwakeword.utils as utils

import microwakeword.test as test
import microwakeword.layers.modes as modes


def compute_false_rates(
    true_positives, true_negatives, false_positives, false_negatives
):
    false_positive_rate = np.float64(false_positives) / (
        false_positives + true_negatives
    )
    false_negative_rate = np.float64(false_negatives) / (
        true_positives + false_negatives
    )

    return false_positive_rate, false_negative_rate


def validate_streaming(flags):
    old_batch_size = flags.batch_size
    utils.convert_model_saved(
        flags,
        "stream_state_internal",
        modes.Modes.STREAM_INTERNAL_STATE_INFERENCE,
        weights_name="last_weights",
    )

    folder_name = "tflite_stream_state_internal"
    file_name = "stream_state_internal.tflite"
    utils.convert_saved_model_to_tflite(
        flags,
        os.path.join(flags.train_dir, "stream_state_internal"),
        os.path.join(flags.train_dir, folder_name),
        file_name,
    )

    flags.batch_size = old_batch_size

    false_accepts_per_hour = test.streaming_model_false_accept_rate(
        flags, folder_name, file_name, "dipco_features.npy"
    )
    (
        accuracy,
        recall,
        precision,
        false_positive_rate,
        false_negative_rate,
    ) = test.tflite_model_accuracy(flags, folder_name, file_name, data_set="validation")

    return {
        "false_accepts_per_hour": false_accepts_per_hour,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def train(flags):
    logging.set_verbosity(flags.verbosity)

    # logging.info(flags)
    model = inception.model(flags)

    utils.save_model_summary(model, flags.train_dir)

    logging.info(model.summary())

    data_processor = input_data.FeatureHandler(
        general_negative_data_dir=flags.general_negative_dir,
        adversarial_negative_data_dir=flags.adversarial_negative_dir,
        positive_data_dir=flags.positive_dir,
    )

    # Setup parameters that dictate each stage of training
    training_steps_list = list(map(int, flags.how_many_training_steps.split(",")))
    learning_rates_list = list(map(float, flags.learning_rate.split(",")))
    general_negative_weight_list = list(
        map(float, flags.general_negative_weight.split(","))
    )
    adversarial_negative_weight_list = list(
        map(float, flags.adversarial_negative_weight.split(","))
    )
    positive_weight_list = list(map(float, flags.positive_weight.split(",")))
    general_negative_probability_list = list(
        map(float, flags.general_negative_probability.split(","))
    )
    positive_probability_list = list(map(float, flags.positive_probability.split(",")))

    with open(os.path.join(flags.train_dir, "flags.txt"), "wt") as f:
        pprint.pprint(flags, stream=f)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.legacy.Adam()

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.FalseNegatives(name="fn"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # configure checkpointer
    checkpoint_directory = os.path.join(flags.train_dir, "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    train_writer = tf.summary.create_file_writer(
        os.path.join(flags.summaries_dir, "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(flags.summaries_dir, "validation")
    )

    training_steps_max = np.sum(training_steps_list)

    best_false_accept_per_hour = 1000.0
    best_fnr = 1.0

    results_deque = deque([])

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                general_negative_weight = general_negative_weight_list[i]
                adversarial_negative_weight = adversarial_negative_weight_list[i]
                positive_weight = positive_weight_list[i]
                general_negative_probability = general_negative_probability_list[i]
                positive_probability = positive_probability_list[i]
                break

        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)

        (
            train_fingerprints,
            train_ground_truth,
            train_sample_weights,
        ) = data_processor.get_data(
            "training",
            batch_size=flags.batch_size,
            features_length=flags.spectrogram_length,
            general_negative_probability=general_negative_probability,
            positive_probability=positive_probability,
            general_negative_weight=general_negative_weight,
            adversarial_negative_weight=adversarial_negative_weight,
            positive_weight=positive_weight,
        )

        result = model.train_on_batch(
            train_fingerprints, train_ground_truth, sample_weight=train_sample_weights
        )

        with train_writer.as_default():
            false_positive_rate, false_negative_rate = compute_false_rates(
                true_positives=result[4],
                false_positives=result[5],
                true_negatives=result[6],
                false_negatives=result[7],
            )

            tf.summary.scalar("loss", result[0], step=training_step)
            tf.summary.scalar("accuracy", result[1], step=training_step)
            tf.summary.scalar("recall", result[2], step=training_step)
            tf.summary.scalar("precision", result[3], step=training_step)
            tf.summary.scalar("fpr", false_positive_rate, step=training_step)
            tf.summary.scalar("fnr", false_negative_rate, step=training_step)

            if not training_step % 25:
                train_writer.flush()

        if len(results_deque) >= 5:
            results_deque.popleft()

        results_deque.append(result)

        if not training_step % 5:
            loss = 0.0
            accuracy = 0.0
            recall = 0.0
            precision = 0.0
            for i in range(0, 5):
                loss += results_deque[i][0]
                accuracy += results_deque[i][1]
                recall += results_deque[i][2]
                precision += results_deque[i][3]

            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    accuracy / 5.0 * 100,
                    recall / 5.0 * 100,
                    precision / 5.0 * 100,
                    loss / 5.0,
                )
            )

        is_last_step = training_step == training_steps_max
        if (training_step % flags.eval_step_interval) == 0 or is_last_step:
            model.save_weights(os.path.join(flags.train_dir, "last_weights"))
            validation_metrics = validate_streaming(flags)
            # (
            #     validation_fingerprints,
            #     validation_ground_truth,
            #     validation_sample_weights,
            # ) = data_processor.get_data(
            #     "validation",
            #     batch_size=flags.batch_size,
            #     features_length=flags.spectrogram_length,
            #     truncation_strategy="truncate_start",
            # )

            # for i in range(0, len(validation_fingerprints), flags.batch_size):
            #     result = model.test_on_batch(
            #         validation_fingerprints[i : i + flags.batch_size],
            #         validation_ground_truth[i : i + flags.batch_size],
            #         reset_metrics=(i == 0),
            #     )

            # loss = result[0]
            # accuracy = result[1]
            # recall = result[2]
            # precision = result[3]
            # true_positives = result[4]
            # false_positives = result[5]
            # true_negatives = result[6]
            # false_negatives = result[7]

            # count = true_positives + false_positives + true_negatives + false_negatives

            # false_positive_rate, false_negative_rate = compute_false_rates(
            #     true_positives, true_negatives, false_positives, false_negatives
            # )

            logging.info(
                "Step %d: Validation accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, fpr = %.2f%%, fnr = %.2f%%, false accepts per hour = %.2f",
                *(
                    training_step,
                    validation_metrics['accuracy'] * 100,
                    validation_metrics['recall'] * 100,
                    validation_metrics['precision'] * 100,
                    validation_metrics['false_positive_rate'] * 100,
                    validation_metrics['false_negative_rate'] * 100,
                    validation_metrics['false_accepts_per_hour'],
                )
            )

            with validation_writer.as_default():
                # tf.summary.scalar("loss", loss, step=training_step)
                tf.summary.scalar("accuracy", validation_metrics['accuracy'] , step=training_step)
                tf.summary.scalar("recall", validation_metrics['recall'], step=training_step)
                tf.summary.scalar("precision", validation_metrics['precision'], step=training_step)
                tf.summary.scalar("fpr", validation_metrics['false_positive_rate'], step=training_step)
                tf.summary.scalar("fnr", validation_metrics['false_negative_rate'], step=training_step)
                tf.summary.scalar("faph", validation_metrics['false_accepts_per_hour'], step=training_step)
                validation_writer.flush()

            model.save_weights(
                os.path.join(
                    flags.train_dir,
                    "train/",
                    str(int(best_false_accept_per_hour * 10000)) + "weights_" + str(training_step),
                )
            )

            false_accepts_rate = validation_metrics['false_accepts_per_hour']
            # Save the model checkpoint when validation accuracy improves
            if (false_accepts_rate < best_false_accept_per_hour) or (
                false_accepts_rate < flags.target_fpr
            ):
                if false_accepts_rate < flags.target_fpr:
                    if (validation_metrics['false_negative_rate'] < best_fnr):
                        best_false_accept_per_hour = false_accepts_rate
                        best_fnr = validation_metrics['false_negative_rate']

                        # overwrite the best model weights
                        model.save_weights(
                            os.path.join(flags.train_dir, "best_weights")
                        )
                        checkpoint.save(file_prefix=checkpoint_prefix)
                else:
                    best_false_accept_per_hour = false_accepts_rate
                    best_fnr = validation_metrics['false_negative_rate']

                    # overwrite the best model weights
                    model.save_weights(os.path.join(flags.train_dir, "best_weights"))
                    checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best low false accepts per hour rate is %.2f with false negative rate of %.2f%%",
                best_false_accept_per_hour,
                (best_fnr * 100),
            )

    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        "testing",
        batch_size=flags.batch_size,
        features_length=flags.spectrogram_length,
        truncation_strategy="truncate_start",
    )

    for i in range(0, len(testing_fingerprints), flags.batch_size):
        result = model.test_on_batch(
            testing_fingerprints[i : i + flags.batch_size],
            testing_ground_truth[i : i + flags.batch_size],
            reset_metrics=(i == 0),
        )
        # summary = tf.Summary(value=[
        #     tf.Summary.Value(tag='accuracy', simple_value=result[1]),])

    accuracy = result[1]
    recall = result[2]
    precision = result[3]
    true_positives = result[4]
    false_positives = result[5]
    true_negatives = result[6]
    false_negatives = result[7]

    count = true_positives + false_positives + true_negatives + false_negatives

    false_positive_rate, false_negative_rate = compute_false_rates(
        true_positives, true_negatives, false_positives, false_negatives
    )

    logging.info(
        "Final testing: accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, fpr = %.2f%%, fnr = %.2f%% (N=%d)",
        *(
            accuracy * 100,
            recall * 100,
            precision * 100,
            false_positive_rate * 100,
            false_negative_rate * 100,
            count,
        )
    )

    with open(os.path.join(flags.train_dir, "metrics_last.txt"), "wt") as fd:
        fd.write(
            "accuracy="
            + str(accuracy * 100)
            + "; recall="
            + str(recall * 100)
            + "; precision="
            + str(precision * 100)
            + "; fpr="
            + str(false_positive_rate * 100)
            + "; fnr="
            + str(false_negative_rate * 100)
        )
    model.save_weights(os.path.join(flags.train_dir, "last_weights"))
