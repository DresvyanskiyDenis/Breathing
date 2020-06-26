import os

import numpy as np
import pandas as pd
import tensorflow as tf

from DDS.utils import create_model, data_instance, average_predictions_by_timesteps

if __name__ == "__main__":
    # params
    path_to_model_0='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_0.h5'
    path_to_model_1='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_1.h5'
    path_to_model_2='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_2.h5'
    path_to_model_3='C:\\Users\\Dresvyanskiy\\Desktop\\Projects\\Breathing\\1d_cnn_models\\best_model_weights_idx_of_part_3.h5'
    path_to_data='C:\\Users\\Dresvyanskiy\\Desktop\\Databases\\DDS\\audio_wav\\'
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
    '''model_2=create_model(input_shape)
    model_2.load_weights(path_to_model_2)
    model_2.compile(optimizer='SGD', loss='mse')
    model_3=create_model(input_shape)
    model_3.load_weights(path_to_model_3)
    model_3.compile(optimizer='SGD', loss='mse')'''
    models=[model_0,model_1]#,model_2,model_3]
    # load data and predict breathing in loop
    files=os.listdir(path_to_data)
    for wavfilename in files:
        # TODO: не правильно выбрал таймстемы - ты сохраняешь их для данных, а нужно для лейблов. labels_rate=25
        instance=data_instance(window_size, window_step)
        instance.load_and_cut_data(path_to_data=path_to_data+wavfilename, cutting_mode='without_padding')
        instance.cutted_data=instance.cutted_data.reshape(instance.cutted_data.shape+(1,))
        prediction_from_different_models=pd.DataFrame(data=np.zeros(shape=(int(instance.data.shape[0]/labels_rate), len(models))),
                                                      columns=['prediction_from_model_'+str(i) for i in range(len(models))])
        for model_idx in range(len(models)):
            predictions=models[model_idx].predict(instance.cutted_data)
            averaged_predictions=average_predictions_by_timesteps(predictions, instance.cutted_data_timesteps)
            prediction_from_different_models.iloc[:, model_idx]=averaged_predictions
        prediction_from_different_models=prediction_from_different_models.mean(axis=1)

