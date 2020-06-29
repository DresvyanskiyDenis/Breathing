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

def create_model(input_shape):
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(input_shape=input_shape, filters=64, kernel_size=10, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=10))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=8, strides=1, activation='relu', padding='same'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=6, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.MaxPool1D(pool_size=4))
    model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5, strides=1, activation='relu', padding='same'       ))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.AvgPool1D(pool_size=4))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='tanh')))
    model.add(tf.keras.layers.Flatten())
    print(model.summary())

    return model

def average_predictions_by_timesteps(predicted_values, timesteps):
    predicted_values=predicted_values.reshape(timesteps.shape)
    result=pd.DataFrame(data=np.concatenate([predicted_values.flatten()[..., np.newaxis], timesteps.flatten()[..., np.newaxis]], axis=1), columns=['predicted_value', 'timestep'])
    tmp=result.groupby(by=['timestep']).mean()
    return tmp


class data_instance():

    def __init__(self, window_size, window_step):
        self.window_size=window_size
        self.window_step=window_step
        self.data=None
        self.cutted_data=None

    def load_data(self, path_to_data, needed_frame_rate=-1):
        self.frame_rate, self.data = wavfile.read(path_to_data)
        print('loading data for %s, original frame rate: %i, needed_frame_rate: %i, ratio: %f' % (path_to_data.split('\\')[-1], self.frame_rate, needed_frame_rate, self.frame_rate / needed_frame_rate))
        if needed_frame_rate!=-1 and needed_frame_rate!=self.frame_rate:
            ratio=self.frame_rate/needed_frame_rate
            if ratio<1: ratio=1./ratio
            ratio=int(ratio)
            self.frame_rate=needed_frame_rate
            print('first size of data:', self.data.shape)
            self.data=self.data[::ratio]
            print('size of data after reduction:', self.data.shape)




    def cut_data_to_length(self, length_to_cut_data):
        self.data=self.data[:length_to_cut_data]

    def cut_data_on_windows(self, cutting_mode, padding_mode=None):
        num_windows=how_many_windows_do_i_need(self.data.shape[0], self.window_size, self.window_step)
        cutted_data=np.zeros(shape=(num_windows, self.window_size))
        start=0
        for idx_window in range(num_windows-1):
            end=start+self.window_size
            cutted_data[idx_window]=self.data[start:end]
            start+=self.window_step
        # last window
        if cutting_mode=='with_padding':
            end=start+self.window_size
            cutted_data[num_windows-1]=self.pad_the_sequence(self.data[start:end], mode=padding_mode)
        elif cutting_mode=='without_padding':
            end=self.data.shape[0]
            start=end-self.window_size
            cutted_data[num_windows-1]=self.data[start:end]
        else:
            raise AttributeError('cutting_mode can be either with_padding or without_padding')
        return cutted_data.astype('float32')

    def load_and_cut_data(self, path_to_data, cutting_mode, padding_mode=None):
        self.load_data(path_to_data)
        self.cutted_data=self.cut_data_on_windows(cutting_mode=cutting_mode, padding_mode=padding_mode)

    def pad_the_sequence(self, sequence, mode, padding_value=0):
        result=np.ones(shape=(self.window_size))*padding_value
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


if __name__ == "__main__":
    path='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\DDS\\audio_wav\\development_01_AUDIO.wav'
    window_size=256000
    window_step=102400
    instance=data_instance(window_size, window_step)
    instance.load_and_cut_data(path, cutting_mode='with_padding', padding_mode='center')
    print(instance.data.shape)
    print(instance.cutted_data.shape)