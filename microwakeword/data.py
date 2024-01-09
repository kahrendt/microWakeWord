import os
import random

import numpy as np

from pathlib import Path
from mmap_ninja.ragged import RaggedMmap

class FeatureHandler(object):
    def __init__(self, general_negative_data_dir, adversarial_negative_data_dir, positive_data_dir):
        self.general_negative_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_general_negative_data_features = []
        self.adversarial_negative_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_adversarial_negative_data_features = []
        self.positive_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_positive_data_features = []
        
        self.prepare_data('general_negative_data', general_negative_data_dir, self.loaded_general_negative_data_features, self.general_negative_data_index)
        self.prepare_data('adversarial_negative', adversarial_negative_data_dir, self.loaded_adversarial_negative_data_features, self.adversarial_negative_data_index)
        self.prepare_data('positive', positive_data_dir, self.loaded_positive_data_features, self.positive_data_index)
         
    def prepare_data(self, type, data_dir, loaded_data_features, data_index):
        if not os.path.exists(data_dir):
            print("ERROR:" + str(type) + "directory doesn't exist")
            
        dirs = ['testing', 'training', 'validation']
        
        for set_index in dirs:
            search_path_directory = os.path.join(data_dir, set_index)
            search_path = [str(i) for i in Path(os.path.abspath(search_path_directory)).glob("**/*_mmap/")]

            for mmap_path in search_path:
                imported_features = RaggedMmap(mmap_path)
                
                loaded_data_features.append(imported_features)
                feature_index = len(loaded_data_features) - 1
                
                for i in range(0, len(imported_features)):
                    data_index[set_index].append({'label': type, 'loaded_feature_index': feature_index, 'subindex': i})

            random.shuffle(data_index[set_index])

    def set_size(self, mode):
        return len(self.data_index[mode])
    
    def get_truncated_features(self, mode, type, index, features_length, truncation_strategy='random'):
        if type == 'background':
            candidates = self.general_negative_data_index[mode]
            sample = candidates[index]
            numpy_features = self.loaded_general_negative_data_features[sample['loaded_feature_index']][sample['subindex']]
        elif type == 'adversarial_negative':
            candidates = self.adversarial_negative_data_index[mode]
            sample = candidates[index]
            numpy_features = self.loaded_adversarial_negative_data_features[sample['loaded_feature_index']][sample['subindex']]            
        elif type == 'positive':
            candidates = self.positive_data_index[mode]   
            sample = candidates[index]
            numpy_features = self.loaded_positive_data_features[sample['loaded_feature_index']][sample['subindex']]  
                 
        data_length = numpy_features.shape[0]
        
        # If the spectrogram is longer than the training shape, we randomly choose a subset
        if (data_length > features_length):
            if truncation_strategy == 'random':
                features_offset = np.random.randint(0, data_length - features_length)
            elif truncation_strategy == 'none':
                # return the entire spectrogram
                features_offset = 0
                features_length = data_length
            elif truncation_strategy == 'truncate_start':
                features_offset = data_length - features_length
            elif truncation_strategy == 'truncate_end':
                features_offset = 0
        else:
            features_offset = 0
        
        return numpy_features[features_offset:(features_offset+features_length)]        
    
    def get_random_features_for_type(self, mode, type, features_length, truncation_strategy):
        if type == 'background':
            data_size = len(self.general_negative_data_index[mode])
        elif type == 'adversarial_negative':
            data_size = len(self.adversarial_negative_data_index[mode])
        elif type == 'positive':
            data_size = len(self.positive_data_index[mode])
            
        sample_index = np.random.randint(0, data_size)
        
        return self.get_truncated_features(mode, type, sample_index, features_length, truncation_strategy=truncation_strategy)

    def get_data(self, mode, batch_size, features_length, general_negative_weight=1.0, adversarial_negative_weight=1.0, positive_weight=1.0, general_negative_probability=0.3, positive_probability=0.5, truncation_strategy='random'):
        general_negative_candidates = self.general_negative_data_index[mode]
        adversarial_negative_candidates = self.adversarial_negative_data_index[mode]
        positive_candidates = self.positive_data_index[mode]
                
        if mode == 'training':
            sample_count = batch_size
        else:
            general_negative_count = len(general_negative_candidates)
            positive_count = len(positive_candidates)
            adversarial_negative_count = len(adversarial_negative_candidates)
            sample_count = general_negative_count + positive_count + adversarial_negative_count

        spectrogram_shape = (features_length, 40)
        data = np.zeros((sample_count,) + spectrogram_shape)
        labels = np.full(sample_count, False)
        weights = np.ones(sample_count)
        
        if mode == 'training':
            random_prob = np.random.rand(sample_count)
            
            for i in range(0, sample_count):
                if random_prob[i] < general_negative_probability:
                    data[i] = self.get_random_features_for_type(mode, 'background', features_length, truncation_strategy=truncation_strategy)
                    labels[i] = False
                    weights[i] = general_negative_weight
                elif random_prob[i] < (general_negative_probability+positive_probability):
                    data[i] = self.get_random_features_for_type(mode, 'positive', features_length, truncation_strategy=truncation_strategy)
                    labels[i] = True
                    weights[i] = positive_weight
                else:
                    data[i] = self.get_random_features_for_type(mode, 'adversarial_negative', features_length, truncation_strategy=truncation_strategy)
                    labels[i] = False
                    weights[i] = adversarial_negative_weight
        else:
            for index in range(len(general_negative_candidates)):
                data[index] = self.get_truncated_features(mode, 'background', index, features_length, truncation_strategy=truncation_strategy)
                labels[index] = False
                weights[index] = general_negative_weight
            for index in range(len(adversarial_negative_candidates)):
                offset_index = index + len(general_negative_candidates)
                
                data[offset_index] = self.get_truncated_features(mode, 'adversarial_negative', index, features_length, truncation_strategy=truncation_strategy)
                labels[offset_index] = False
                weights[offset_index] = adversarial_negative_weight
            for index in range(len(positive_candidates)):
                offset_index = index + len(general_negative_candidates) + len(adversarial_negative_candidates)
                
                data[offset_index] = self.get_truncated_features(mode, 'positive', index, features_length, truncation_strategy=truncation_strategy)
                labels[offset_index] = True
                weights[offset_index] = positive_weight
            
            # Randomize the order of the testing and validation sets    
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            
            data = data[indices]
            labels = labels[indices]
            weights = weights[indices]
        return data, labels, weights
    