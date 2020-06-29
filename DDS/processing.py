import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import wavfile

from DDS.utils import create_model, data_instance, average_predictions_by_timesteps

if __name__ == "__main__":
    # params
    path_to_model_0='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_0.h5'
    path_to_model_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_1.h5'
    path_to_model_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_2.h5'
    path_to_model_3='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_3.h5'
    path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\DDS\\audio_wav\\'
    path_to_save_predictions='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\DDS\\predictions\\'
    if not os.path.exists(path_to_save_predictions):
        os.mkdir(path_to_save_predictions)
    labels_rate=25
    window_size=256000
    window_step=102400
    input_shape=(window_size, 1)
    # create models and load corresponding weights
    model_0=create_model(input_shape)
    model_0.load_weights(path_to_model_0)
    model_0.compile(optimizer='SGD', loss='mse')
    model_1=create_model(input_shape)
    model_1.load_weights(path_to_model_1)
    model_1.compile(optimizer='SGD', loss='mse')
    model_2=create_model(input_shape)
    model_2.load_weights(path_to_model_2)
    model_2.compile(optimizer='SGD', loss='mse')
    model_3=create_model(input_shape)
    model_3.load_weights(path_to_model_3)
    model_3.compile(optimizer='SGD', loss='mse')
    models=[model_0,model_1,model_2,model_3]
    # load data and predict breathing in loop
    files=os.listdir(path_to_data)
    counter=0
    for wavfilename in files:
        # data
        start=time.time()
        data_inst=data_instance(window_size, window_step)
        data_inst.load_data(path_to_data=path_to_data+wavfilename, needed_frame_rate=16000)
        predictions_full_length = int(data_inst.data.shape[0] / data_inst.frame_rate * labels_rate)                     # evaluate predictions length
        data_inst.cut_data_to_length(length_to_cut_data=int(predictions_full_length/labels_rate*data_inst.frame_rate))  # now we cutted data to predictions size
        data_inst.cutted_data=data_inst.cut_data_on_windows(cutting_mode='without_padding')
        data_inst.cutted_data=data_inst.cutted_data.reshape(data_inst.cutted_data.shape+(1,))
        # timesteps for averaging the predictions (because of windows and step of window)
        timesteps=data_instance(int(window_size/data_inst.frame_rate*labels_rate), int(window_step/data_inst.frame_rate*labels_rate))
        timesteps.frame_rate=labels_rate
        timesteps.data=np.array([i*(1./labels_rate) for i in range(predictions_full_length)])
        timesteps.cutted_data=timesteps.cut_data_on_windows(cutting_mode='without_padding')

        prediction_from_different_models=pd.DataFrame(data=np.zeros(shape=(predictions_full_length, 1+len(models))),
                                                      columns=['timestep']+['prediction_from_model_'+str(i) for i in range(len(models))])
        prediction_from_different_models['timestep']=timesteps.data.copy()
        prediction_from_different_models.set_index(['timestep'], inplace=True)

        for model_idx in range(len(models)):
            predictions=models[model_idx].predict(data_inst.cutted_data)
            averaged_predictions=average_predictions_by_timesteps(predictions, timesteps.cutted_data)
            prediction_from_different_models.iloc[:, model_idx]=averaged_predictions.values
        # calculate mean of 4 predictions
        prediction_from_different_models=pd.DataFrame(prediction_from_different_models.mean(axis=1))
        # format data to write in csv file
        prediction_from_different_models=prediction_from_different_models.reset_index()
        prediction_from_different_models['timestep']=prediction_from_different_models['timestep'].apply(lambda x: round(x, 2))
        prediction_from_different_models.set_index(['timestep'], inplace=True)
        # save predictions to csv file
        prediction_from_different_models.to_csv(path_to_save_predictions+wavfilename.split('.')[0]+'.csv')
        counter+=1
        print('filename %s processed... remaining: %i   processing time: %f'%(wavfilename,len(files)-counter, time.time()-start))


