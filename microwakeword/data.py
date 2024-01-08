import os
import random

import numpy as np

from pathlib import Path
from mmap_ninja.ragged import RaggedMmap

class FeatureHandler(object):
    def __init__(self, background_data_dir, generated_negative_data_dir, generated_positive_data_dir):
        # self.data_index = {'validation': [], 'testing': [], 'training': []}
        # self.loaded_data_features = []
        self.background_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_background_data_features = []
        self.generated_negative_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_generated_negative_data_features = []
        self.generated_positive_data_index = {'validation': [], 'testing': [], 'training': []}
        self.loaded_generated_positive_data_features = []
        
        self.prepare_data('background_data', background_data_dir, self.loaded_background_data_features, self.background_data_index)
        self.prepare_data('generated_negative', generated_negative_data_dir, self.loaded_generated_negative_data_features, self.generated_negative_data_index)
        self.prepare_data('generated_positive', generated_positive_data_dir, self.loaded_generated_positive_data_features, self.generated_positive_data_index)
         
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
                # self.loaded_data_features.append(imported_features)
                feature_index = len(loaded_data_features) - 1
                # feature_index = len(self.loaded_data_features) - 1
                
                for i in range(0, len(imported_features)):
                    data_index[set_index].append({'label': type, 'loaded_feature_index': feature_index, 'subindex': i})
                    # self.data_index[set_index].append({'label': type, 'loaded_feature_index': feature_index, 'subindex': i})
                    
            random.shuffle(data_index[set_index])
            # random.shuffle(self.data_index[set_index])

    def set_size(self, mode):
        return len(self.data_index[mode])
    
    def get_truncated_features(self, mode, type, index, features_length):
        if type == 'background':
            candidates = self.background_data_index[mode]
            sample = candidates[index]
            numpy_features = self.loaded_background_data_features[sample['loaded_feature_index']][sample['subindex']]
        elif type == 'generated_negative':
            candidates = self.generated_negative_data_index[mode]
            sample = candidates[index]
            numpy_features = self.loaded_generated_negative_data_features[sample['loaded_feature_index']][sample['subindex']]            
        elif type == 'generated_positive':
            candidates = self.generated_positive_data_index[mode]   
            sample = candidates[index]
            numpy_features = self.loaded_generated_positive_data_features[sample['loaded_feature_index']][sample['subindex']]  
                 
        data_length = numpy_features.shape[0]
        
        # If the spectrogram is longer than the training shape, we randomly choose a subset
        if (data_length > features_length):
            features_offset = np.random.randint(0, data_length - features_length)
        else:
            features_offset = 0
        
        return numpy_features[features_offset:(features_offset+features_length)]        
    
    def get_random_features_for_type(self, mode, type, features_length):
        if type == 'background':
            data_size = len(self.background_data_index[mode])
        elif type == 'generated_negative':
            data_size = len(self.generated_negative_data_index[mode])
        elif type == 'generated_positive':
            data_size = len(self.generated_positive_data_index[mode])
            
        sample_index = np.random.randint(0, data_size)
        
        return self.get_truncated_features(mode, type, sample_index, features_length)

    def get_data(self, mode, batch_size, features_length, background_weight=1.0, generated_negative_weight=1.0, generated_positive_weight=1.0, background_probability=0.3, positive_probability=0.5):
        # candidates = self.data_index[mode]
        
        # spectrogram_shape = (features_length, 40)
        # if mode == 'training':
        #     sample_count = batch_size
        # else:
        #     sample_count = len(candidates)
        
        # data = np.zeros((sample_count,) + spectrogram_shape)
        # labels = np.full(sample_count, False)
        # weights = np.ones(sample_count)   
        
        # for i in range(0, sample_count):
        #     if mode == 'training':
        #         sample_index = np.random.randint(0, len(candidates))
        #     else:
        #         sample_index = i
            
        #     sample = candidates[sample_index]
            
        #     if sample['label'] == 'generated_positive':
        #         sample_truth = True
        #         sample_weight = generated_positive_weight
        #     elif sample['label'] == 'generated_negative':
        #         sample_truth = False
        #         sample_weight = generated_negative_weight
        #     elif sample['label'] == 'background_data':
        #         sample_truth = False
        #         sample_weight = background_weight
            
        #     numpy_features = self.loaded_data_features[sample['loaded_feature_index']][sample['subindex']]
            
        #     data_length = numpy_features.shape[0]
            
        #     if (data_length > features_length):
        #         features_offset = np.random.randint(0, data_length - features_length)
        #     else:
        #         features_offset = 0
            
        #     clipped_data = numpy_features[features_offset:(features_offset+features_length)]
            
        #     data[i] = clipped_data
        #     labels[i] = sample_truth
        #     weights[i] = sample_weight
            
        # return data, labels, weights
        
        background_candidates = self.background_data_index[mode]
        generated_negative_candidates = self.generated_negative_data_index[mode]
        generated_positive_candidates = self.generated_positive_data_index[mode]
                
        if mode == 'training':
            sample_count = batch_size
        else:
            background_count = len(background_candidates)
            generated_positive_count = len(generated_positive_candidates)
            generated_negative_count = len(generated_negative_candidates)
            sample_count = background_count + generated_positive_count + generated_negative_count

        spectrogram_shape = (features_length, 40)
        data = np.zeros((sample_count,) + spectrogram_shape)
        labels = np.full(sample_count, False)
        weights = np.ones(sample_count)
        
        if mode == 'training':
            random_prob = np.random.rand(sample_count)
            
            for i in range(0, sample_count):
                if random_prob[i] < background_probability:
                    data[i] = self.get_random_features_for_type(mode, 'background', features_length)
                    labels[i] = False
                    weights[i] = background_weight
                elif random_prob[i] < (background_probability+positive_probability):
                    data[i] = self.get_random_features_for_type(mode, 'generated_positive', features_length)
                    labels[i] = True
                    weights[i] = generated_positive_weight
                else:
                    data[i] = self.get_random_features_for_type(mode, 'generated_negative', features_length)
                    labels[i] = False
                    weights[i] = generated_negative_weight
        else:
            for index in range(len(background_candidates)):
                data[index] = self.get_truncated_features(mode, 'background', index, features_length)
                labels[index] = False
                weights[index] = background_weight
            for index in range(len(generated_negative_candidates)):
                offset_index = index + len(background_candidates)
                
                data[offset_index] = self.get_truncated_features(mode, 'generated_negative', index, features_length)
                labels[offset_index] = False
                weights[offset_index] = generated_negative_weight
            for index in range(len(generated_positive_candidates)):
                offset_index = index + len(background_candidates) + len(generated_negative_candidates)
                
                data[offset_index] = self.get_truncated_features(mode, 'generated_positive', index, features_length)
                labels[offset_index] = True
                weights[offset_index] = generated_positive_weight
            
            # Randomize the order of the testing and validation sets    
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            
            data = data[indices]
            labels = labels[indices]
            weights = weights[indices]
        return data, labels, weights
    
        for i in range(0, background_count):
            if mode == 'training':
                sample_index = np.random.randint(0, len(background_candidates))
            else:
                sample_index = i

            background_sample = background_candidates[sample_index]
            
            numpy_features = self.loaded_background_data_features[background_sample['loaded_feature_index']][background_sample['subindex']]
            
            data_length = numpy_features.shape[0]
            
            # Spectrogram is longer than the training shape, so randomly choose a subset
            if (data_length > features_length):
                features_offset = np.random.randint(0, data_length - features_length)
            else:
                features_offset = 0
            
            clipped_data = numpy_features[features_offset:(features_offset+features_length)]
    
            data[i] = clipped_data
            labels[i] = False
            weights[i] = background_weight
            
        for i in range(0, generated_positive_count):
        # for i in range(background_count, background_count + generated_positive_count):
            if mode == 'training':
                sample_index = np.random.randint(0, len(generated_positive_candidates))
            else:
                sample_index = i
            
            positive_sample = generated_positive_candidates[sample_index]
            
            numpy_features = self.loaded_generated_positive_data_features[positive_sample['loaded_feature_index']][positive_sample['subindex']]
            
            data_length = numpy_features.shape[0]
            
            # Spectrogram is longer than the training shape, so randomly choose a subset
            if (data_length > features_length):
                features_offset = np.random.randint(0, data_length - features_length)
            else:
                features_offset = 0
            
            clipped_data = numpy_features[features_offset:(features_offset+features_length)]
    
            data[i+background_count] = clipped_data
            labels[i+background_count] = True
            weights[i+background_count] = generated_positive_weight

        for i in range(0, generated_negative_count):
        # for i in range(background_count + generated_positive_count, background_count+generated_positive_count+generated_negative_count):
            if mode == 'training':
                sample_index = np.random.randint(0, len(generated_negative_candidates))
            else:
                sample_index = i
                
            negative_sample = generated_negative_candidates[sample_index]
            
            numpy_features = self.loaded_generated_negative_data_features[negative_sample['loaded_feature_index']][negative_sample['subindex']]
            
            data_length = numpy_features.shape[0]
            
            # Spectrogram is longer than the training shape, so randomly choose a subset
            if (data_length > features_length):
                features_offset = np.random.randint(0, data_length - features_length)
            else:
                features_offset = 0
            
            clipped_data = numpy_features[features_offset:(features_offset+features_length)]
                
            data[i+background_count+generated_positive_count] = clipped_data
            labels[i+background_count+generated_positive_count] = False
            weights[i+background_count+generated_positive_count] = generated_negative_weight
        
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        
        data = data[indices]
        labels = labels[indices]

        return data, labels, weights    
        # complete_batches_count = sample_count//batch_size
        # return data[0:complete_batches_count*batch_size], labels[0:complete_batches_count*batch_size]