import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import keras

class InverseModelDense():
    def __init__(self, main_model, tracking_layer):
        self.main_model = main_model
        self.tracking_layer = tracking_layer

        self.aux_model = self.create_aux_model()

    def create_aux_model(self):
        aux_model = tf.keras.models.Model(inputs=self.main_model.inputs,
                                        outputs=self.main_model.outputs + [self.tracking_layer.output])
        return aux_model

    def dimension_reduction(self, X, dim):
        if (dim < 50):
            if (X.shape[-1] > 50):
                pca = PCA(n_components=50)
                pca.fit(X)
                X = pca.transform(X)

            tsne = TSNE(n_components=dim, learning_rate="auto")
            X_comp = tsne.fit_transform(X)

        else:
            pca = PCA(n_components=dim)
            X_comp = pca.fit_transform(X)

        return X_comp


    def sample(self, inputs, dimensions=[3]):
        layer_out = self.aux_model.predict(inputs)[-1]

        x_dim = {}

        for dim in dimensions:
            x_dim[dim] = self.dimension_reduction(layer_out, dim)


        return x_dim
