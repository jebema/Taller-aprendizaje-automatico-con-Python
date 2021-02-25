# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:01:57 2019

@author: jessi
"""

import os
import argparse 

import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from python_speech_features import mfcc
#from sklearn.externals import joblib
import joblib

# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    hmm_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []

        # Iterate through the audio files 
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)

            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            # Append the label
            y_words.append(label)

        print ('X.shape =', X.shape)
        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        
        joblib.dump(hmm_models, "HMMmodels.pkl")



