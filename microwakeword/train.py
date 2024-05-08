# coding=utf-8
# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from absl import logging

import numpy as np
import tensorflow as tf

import microwakeword.test as test


def validate_nonstreaming(config, data_processor, model, test_set):
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )

    test_batch_size = 1024
    
    model.reset_metrics()
    for i in range(0, len(testing_fingerprints), test_batch_size):
        result = model.test_on_batch(
            testing_fingerprints[i : i + test_batch_size],
            testing_ground_truth[i : i + test_batch_size],
            reset_metrics=False,
        )

    true_positives = result[4]
    false_positives = result[5]
    true_negatives = result[6]
    false_negatives = result[7]

    metrics = test.compute_metrics(
        true_positives, true_negatives, false_positives, false_negatives
    )

    metrics["loss"] = result[9]
    metrics["auc"] = result[8]

    ambient_false_positives = 0  # float("nan")
    estimated_ambient_false_positives_per_hour = 0  # float("nan")
    
    recall_at_no_faph = 0

    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        model.reset_metrics()
        
        cutoffs = np.arange(0.5,1.01,0.01)
        batch_sum_false_positives = np.zeros(cutoffs.shape[0])
        
        for i in range(0, len(ambient_testing_fingerprints), test_batch_size):
            ambient_predictions = model.predict_on_batch(
                ambient_testing_fingerprints[i : i + test_batch_size]
            )
            
            for index, cutoff in enumerate(cutoffs):
                batch_sum_false_positives[index] += sum(ambient_predictions > cutoff)
            # ambient_result = model.test_on_batch(
            #     ambient_testing_fingerprints[i : i + test_batch_size],
            #     ambient_testing_ground_truth[i : i + test_batch_size],
            #     reset_metrics=False,
            # )

        # estimated_ambient_false_positives_per_hour = ambient_false_positives / (
        #     data_processor.get_mode_duration("validation_ambient") / 3600.0
        # )

        batch_sum_false_positives_per_hour = batch_sum_false_positives/ (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        ambient_false_positives = batch_sum_false_positives[0]
        estimated_ambient_false_positives_per_hour = batch_sum_false_positives_per_hour[0]
        
        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(cutoffs):
            if batch_sum_false_positives_per_hour[index] == 0:
                target_faph_cutoff_probability = cutoff
                break
        
        if target_faph_cutoff_probability < 1.0:
            total_positive_sample_count = 0
            total_predicted_at_cutoff = 0
            for i in range(0, len(testing_fingerprints), test_batch_size):
                predictions = model.predict_on_batch(
                    testing_fingerprints[i : i + test_batch_size]
                )
                #     testing_ground_truth[i : i + test_batch_size],
                #     reset_metrics=False,
                # )        
                total_positive_sample_count += sum(testing_ground_truth[i : i + test_batch_size])
                total_predicted_at_cutoff += sum(predictions[testing_ground_truth[i : i + test_batch_size].nonzero()] > target_faph_cutoff_probability)
            
            recall_at_no_faph = total_predicted_at_cutoff[0]/total_positive_sample_count
            # print(total_positive_sample_count, total_predicted_at_cutoff[0])
            # print("Recall at 1 faph:", total_predicted_at_cutoff[0]/total_positive_sample_count, "cutoff level is", target_faph_cutoff_probability)

        

    metrics["recall_at_no_faph"] = recall_at_no_faph
    metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
    metrics["ambient_false_positives"] = ambient_false_positives
    metrics["ambient_false_positives_per_hour"] = (
        estimated_ambient_false_positives_per_hour
    )

    return metrics


def train(model, config, data_processor):

    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

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
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="loss"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }

        (
            train_fingerprints,
            train_ground_truth,
            train_sample_weights,
        ) = data_processor.get_data(
            "training",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="default",
            augmentation_policy=augmentation_policy,
        )

        class_weights = {0: negative_class_weight, 1: positive_class_weight}

        result = model.train_on_batch(
            train_fingerprints,
            train_ground_truth,
            sample_weight=train_sample_weights,
            class_weight=class_weights,
            reset_metrics=False,
        )

        # Print the running statistics in the current validation epoch
        print("Validation Batch #{:d}: Accuracy = {:.3f}; Recall = {:.3f}; Precision = {:.3f}; Loss = {:.4f}; Mini-Batch #{:d}".format((training_step//config["eval_step_interval"]+1), result[1], result[2], result[3], result[9], (training_step % config["eval_step_interval"])), end='\r')

        is_last_step = training_step == training_steps_max
        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            logging.info(
                "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                *(
                    training_step,
                    learning_rate,
                    result[1]*100,
                    result[2]* 100,
                    result[3] * 100,
                    result[9],
                ),
            )
            
            metrics = test.compute_metrics(
                true_positives=result[4],
                false_positives=result[5],
                true_negatives=result[6],
                false_negatives=result[7],
            )     
            
            with train_writer.as_default():
                tf.summary.scalar("loss", result[9], step=training_step)
                tf.summary.scalar("accuracy", result[1], step=training_step)
                tf.summary.scalar("recall", result[2], step=training_step)
                tf.summary.scalar("precision", result[3], step=training_step)
                tf.summary.scalar("fpr", metrics["false_positive_rate"], step=training_step)
                tf.summary.scalar("fnr", metrics["false_negative_rate"], step=training_step)
                tf.summary.scalar("auc", result[8], step=training_step)   
                train_writer.flush()     
            
            
            model.save_weights(os.path.join(config["train_dir"], "last_weights"))

            nonstreaming_metrics = validate_nonstreaming(
                config, data_processor, model, "validation"
            )
            model.reset_metrics()   # reset metrics for next validation epoch of training
            logging.info(
                "Step %d (nonstreaming): Validation: recall at no faph = %.3f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, fpr = %.2f%%, fnr = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f,",
                *(
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["false_positive_rate"] * 100,
                    nonstreaming_metrics["false_negative_rate"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                ),
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "fpr",
                    nonstreaming_metrics["false_positive_rate"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "fnr",
                    nonstreaming_metrics["false_negative_rate"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "faph",
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                validation_writer.flush()

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train/",
                    str(int(best_minimization_quantity * 10000))
                    + "weights_"
                    + str(training_step),
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[
                    config["minimization_metric"]
                ]
            current_maximization_quantity = nonstreaming_metrics[
                config["maximization_metric"]
            ]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model
            if (
                (
                    (
                        current_minimization_quantity <= config["target_minimization"]
                    )  # achieved target false positive rate
                    and (
                        (
                            current_maximization_quantity > best_maximization_quantity
                        )  # either accuracy improved
                        or (
                            best_minimization_quantity > config["target_minimization"]
                        )  # or this is the first time we met the target
                    )
                )
                or (
                    (
                        current_minimization_quantity > config["target_minimization"]
                    )  # we haven't achieved our target
                    and (
                        current_minimization_quantity < best_minimization_quantity
                    )  # but we have decreased since the previous best
                )
                or (
                    (
                        current_minimization_quantity == best_minimization_quantity
                    )  # we tied a previous best
                    and (
                        current_maximization_quantity > best_maximization_quantity
                    )  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(os.path.join(config["train_dir"], "best_weights"))
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)

    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        "testing",
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )

    model.reset_metrics()
    for i in range(0, len(testing_fingerprints), config["batch_size"]):
        result = model.test_on_batch(
            testing_fingerprints[i : i + config["batch_size"]],
            testing_ground_truth[i : i + config["batch_size"]],
            reset_metrics=False,
        )

    true_positives = result[4]
    false_positives = result[5]
    true_negatives = result[6]
    false_negatives = result[7]

    metrics = test.compute_metrics(
        true_positives, true_negatives, false_positives, false_negatives
    )
    metrics_string = test.metrics_to_string(metrics)

    logging.info("Last weights on testing set: " + metrics_string)

    with open(os.path.join(config["train_dir"], "metrics_last.txt"), "wt") as fd:
        fd.write(metrics_string)
    model.save_weights(os.path.join(config["train_dir"], "last_weights"))
