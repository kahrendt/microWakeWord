import os
import pprint

from absl import logging
from collections import deque

import numpy as np
import tensorflow as tf

import microwakeword.data as input_data
import microwakeword.inception as inception
import microwakeword.utils as utils

def compute_false_rates(true_positives, true_negatives, false_positives, false_negatives):
    false_positive_rate = np.float64(false_positives)/(false_positives+true_negatives)
    false_negative_rate = np.float64(false_negatives)/(true_positives+false_negatives)
    
    return false_positive_rate, false_negative_rate

def train(flags):
    logging.set_verbosity(flags.verbosity)
    
    # logging.info(flags)
    model = inception.model(flags)

    utils.save_model_summary(model, flags.train_dir)
    
    logging.info(model.summary())    
    
    data_processor = input_data.FeatureHandler(background_data_dir=flags.background_dir, generated_negative_data_dir=flags.generated_negative_dir, generated_positive_data_dir=flags.generated_positive_dir)
    
    # Setup parameters that dictate each stage of training
    training_steps_list = list(map(int, flags.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, flags.learning_rate.split(',')))
    background_weight_list = list(map(float, flags.background_weight.split(',')))
    generated_negative_weight_list = list(map(float, flags.generated_negative_weight.split(',')))
    generated_positive_weight_list = list(map(float, flags.generated_positive_weight.split(',')))
    background_probability_list = list(map(float,flags.background_probability.split(',')))
    positive_probability_list = list(map(float,flags.positive_probability.split(',')))
    

    with open(os.path.join(flags.train_dir, 'flags.txt'), 'wt') as f:
        pprint.pprint(flags, stream=f)
        
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.legacy.Adam()
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
        tf.keras.metrics.Recall(name='recall'), 
        tf.keras.metrics.Precision(name='precision'), 
        tf.keras.metrics.TruePositives(name='tp'), 
        tf.keras.metrics.FalsePositives(name='fp'), 
        tf.keras.metrics.TrueNegatives(name='tn'), 
        tf.keras.metrics.FalseNegatives(name='fn')
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # configure checkpointer
    checkpoint_directory = os.path.join(flags.train_dir, 'restore/')
    checkpoint_prefix = os.path.join(checkpoint_directory, 'ckpt')
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))
    
    train_writer = tf.summary.create_file_writer(os.path.join(flags.summaries_dir, 'train'))
    validation_writer = tf.summary.create_file_writer(os.path.join(flags.summaries_dir, 'validation'))

    training_steps_max = np.sum(training_steps_list)
    
    best_fpr = 1.0
    best_fnr = 1.0
    
    results_deque = deque([])
     
    for training_step in range(1, training_steps_max+1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                background_weight = background_weight_list[i]
                generated_negative_weight = generated_negative_weight_list[i]
                generated_positive_weight = generated_positive_weight_list[i]
                background_probability = background_probability_list[i]
                positive_probability = positive_probability_list[i]
                break

        tf.keras.backend.set_value(model.optimizer.lr, learning_rate)

        train_fingerprints, train_ground_truth, train_sample_weights = data_processor.get_data(
            'training', batch_size=flags.batch_size, features_length=flags.spectrogram_length, background_probability=background_probability, positive_probability=positive_probability, background_weight=background_weight, generated_negative_weight=generated_negative_weight, generated_positive_weight=generated_positive_weight)
        
        result = model.train_on_batch(train_fingerprints, train_ground_truth, sample_weight=train_sample_weights)

        with train_writer.as_default():
            false_positive_rate, false_negative_rate = compute_false_rates(true_positives=result[4], false_positives=result[5], true_negatives=result[6], false_negatives=result[7])
            
            tf.summary.scalar('loss', result[0], step=training_step)
            tf.summary.scalar('accuracy', result[1], step=training_step)
            tf.summary.scalar('recall', result[2], step=training_step)
            tf.summary.scalar('precision', result[3], step=training_step)
            tf.summary.scalar('fpr', false_positive_rate, step=training_step)
            tf.summary.scalar('fnr', false_negative_rate, step=training_step)
            
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
            for i in range(0,5):
                loss += results_deque[i][0]
                accuracy += results_deque[i][1]
                recall += results_deque[i][2]
                precision += results_deque[i][3]
                
            logging.info(
                'Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f',
                *(training_step, learning_rate, accuracy/5.0 * 100, recall/5.0*100, precision/5.0*100, loss/5.0))

        is_last_step = (training_step == training_steps_max)
        if (training_step % flags.eval_step_interval) == 0 or is_last_step:
            validation_fingerprints, validation_ground_truth, validation_sample_weights = data_processor.get_data('validation', batch_size=flags.batch_size, features_length=flags.spectrogram_length)

            for i in range(0, len(validation_fingerprints), flags.batch_size):
                result = model.test_on_batch(validation_fingerprints[i : i+flags.batch_size], validation_ground_truth[i : i+flags.batch_size], reset_metrics=(i==0))
                
            loss = result[0]
            accuracy = result[1]
            recall = result[2]
            precision = result[3]
            true_positives = result[4]
            false_positives = result[5]
            true_negatives = result[6]
            false_negatives = result[7]
            
            count = true_positives + false_positives + true_negatives + false_negatives

            false_positive_rate, false_negative_rate = compute_false_rates(true_positives, true_negatives, false_positives, false_negatives)
            
            logging.info('Step %d: Validation accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, fpr = %.2f%%, fnr = %.2f%% (N=%d)',
                        *(training_step, accuracy * 100, recall * 100, precision * 100, false_positive_rate * 100, false_negative_rate * 100, count))
            
            with validation_writer.as_default():
                tf.summary.scalar('loss', loss, step=training_step)
                tf.summary.scalar('accuracy', accuracy, step=training_step)
                tf.summary.scalar('recall', recall, step=training_step)
                tf.summary.scalar('precision', precision, step=training_step)
                tf.summary.scalar('fpr', false_positive_rate, step=training_step)
                tf.summary.scalar('fnr', false_negative_rate, step=training_step)                
                validation_writer.flush()

            model.save_weights(
                os.path.join(
                    flags.train_dir, 'train/',
                    str(int(best_fpr * 10000)) + 'weights_' +
                    str(training_step)))

            # Save the model checkpoint when validation accuracy improves
            if (false_positive_rate < best_fpr) or (false_positive_rate < 0.025):
                if false_positive_rate < 0.025:
                    if false_negative_rate < best_fnr:
                        best_fpr = false_positive_rate
                        best_fnr = false_negative_rate

                        # overwrite the best model weights
                        model.save_weights(os.path.join(flags.train_dir, 'best_weights'))
                        checkpoint.save(file_prefix=checkpoint_prefix)
                else:
                    best_fpr = false_positive_rate
                    best_fnr = false_negative_rate

                    # overwrite the best model weights
                    model.save_weights(os.path.join(flags.train_dir, 'best_weights'))
                    checkpoint.save(file_prefix=checkpoint_prefix)                 

            logging.info('So far the best low false positive rate is %.2f%% with false negative rate of %.2f%%',
                        (best_fpr * 100), (best_fnr * 100))

    testing_fingerprints, testing_ground_truth, testing_sample_weights = data_processor.get_data('testing', batch_size=flags.batch_size, features_length=flags.spectrogram_length)

    for i in range(0, len(validation_fingerprints), flags.batch_size):
        result = model.test_on_batch(testing_fingerprints[i: i+flags.batch_size], testing_ground_truth[i : i+flags.batch_size], reset_metrics=(i==0))
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

    false_positive_rate, false_negative_rate = compute_false_rates(true_positives, true_negatives, false_positives, false_negatives)
    
    logging.info('Final testing: accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, fpr = %.2f%%, fnr = %.2f%% (N=%d)',
                    *(accuracy * 100, recall * 100, precision * 100, false_positive_rate * 100, false_negative_rate * 100, count))


    with open(os.path.join(flags.train_dir, 'metrics_last.txt'), 'wt') as fd:
        fd.write("accuracy=" + str(accuracy * 100) + "; recall=" + str(recall*100) + "; precision=" + str(precision*100) + "; fpr=" + str(false_positive_rate*100) + "; fnr=" + str(false_negative_rate*100))
    model.save_weights(os.path.join(flags.train_dir, 'last_weights'))