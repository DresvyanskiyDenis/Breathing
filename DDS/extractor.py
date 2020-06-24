import gc

import tensorflow as tf

class Extractor():
    def __init__(self, mode, model, layer_idx=-1):
        self.mode=mode
        self.model=model
        self.layer_idx=layer_idx


    def extract(self, data):
        if self.mode=='features':
            new_model=tf.keras.Model(inputs=self.model.inputs, outputs=[self.model.layers[self.layer_idx]])
            new_model.compile(optimizer=self.model.optimizer, loss=self.model.loss)
            extracted_features=new_model.predict(data)
            del new_model
            tf.keras.backend.clear_session()
            gc.collect()
            return extracted_features
        elif self.mode=='labels':
            predictions=self.model.predict(data)
            return predictions
        else:
            raise AttributeError('mode can be only features or labels')

    def set_model(self, model):
        self.model=model
