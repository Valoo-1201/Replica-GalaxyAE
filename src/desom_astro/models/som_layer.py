"""Capa SOM (Self-Organizing Map) para un DESOM.

Recibe vectores latentes (B, latent_dim) y devuelve la distancia al cuadrado
a cada prototipo (B, n_prototipos). El nodo ganador (BMU) es el argmin.
"""
import tensorflow as tf
from keras.layers import Layer, InputSpec


class SOMLayer(Layer):
    def __init__(self, map_size, prototypes=None, **kwargs):
        super().__init__(**kwargs)
        self.map_size = map_size
        self.n_prototypes = map_size[0] * map_size[1]
        self.initial_prototypes = prototypes
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.prototypes = self.add_weight(
            shape=(self.n_prototypes, input_dim),
            initializer='glorot_uniform',
            name='prototypes',
        )
        if self.initial_prototypes is not None:
            self.set_weights(self.initial_prototypes)
            del self.initial_prototypes
        self.built = True

    def call(self, inputs, **kwargs):
        # (B,1,D) - (K,D) -> (B,K,D); se eleva al cuadrado y se suma sobre D -> (B,K)
        return tf.reduce_sum(
            tf.square(tf.expand_dims(inputs, axis=1) - self.prototypes), axis=2
        )

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_prototypes

    def get_config(self):
        config = {'map_size': self.map_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
