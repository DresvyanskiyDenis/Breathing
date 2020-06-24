import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

def how_many_windows_do_i_need(length_sequence, window_size, step):
    start_idx=0
    counter=0
    while True:
        if start_idx+window_size>length_sequence:
            break
        start_idx+=step
        counter+=1
    if start_idx!=length_sequence:
        counter+=1
    return counter


class data_instance():

    def __init__(self, window_size, window_step):
        self.window_size=window_size
        self.window_step=window_step
        self.data=None

    def load_data(self, path_to_data):
        self.frame_rate, self.data = wavfile.read(path_to_data)

    def cut_data_on_windows(self, mode):
        num_windows=how_many_windows_do_i_need(self.data.shape[0], self.window_size, self.window_step)
        cutted_data=np.zeros(shape=(num_windows, self.window_size))
        start=0
        for idx_window in range(num_windows-1):
            end=start+self.window_size
            cutted_data[idx_window]=self.data[start:end]
            start+=self.window_step
        # last window
        if mode=='with_padding':
            end=start+self.window_size
            cutted_data[num_windows-1]=self.pad_the_sequence(self.data[start:end], mode='center')
        elif mode=='without_padding':
            end=self.data.shape[0]
            start=end-self.window_size
            cutted_data[num_windows-1]=self.data[start:end]
        else:
            raise AttributeError('mode can be either with_padding or without_padding')
        return cutted_data

    def load_and_cut_data(self, path_to_data, mode):
        self.load_data(path_to_data)
        self.cutted_data=self.cut_data_on_windows(mode=mode)

    def pad_the_sequence(self, sequence, mode):
        result=np.zeros(shape=(self.window_size))
        if mode=='left':
            result[(self.window_size-sequence.shape[0]):]=sequence
        elif mode=='right':
            result[:sequence.shape[0]]=sequence
        elif mode=='center':
            start=(self.window_size-sequence.shape[0])//2
            end=start+sequence.shape[0]
            result[start:end]=sequence
        else:
            raise AttributeError('mode can be either left, right or center')
        return result

    def get_data(self):
        return self.data
