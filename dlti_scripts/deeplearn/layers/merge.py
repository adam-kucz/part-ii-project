from funcy import first, second
from tensorflow.keras.layers import Layer


class PartialAverage(Layer):
    def __init__(self, *args, **kwargs):
        


# class LambdaMask(Layer):
#     def __init__(self, func, *args, **kwargs):
#         self.func = func
#         self.supports_masking = True
#         super().__init__(*args, **kwargs)

#     def compute_mask(self, x, mask=None):
#         return second(self.func(x, mask))

#     def call(self, x, mask=None):
#         return first(self.func(x, mask))


# class PartialAverage(LambdaMask):
#     def __init__(self, *args, **kwargs):
#         func = lambda x, mask: (tf.keras.x, mask)
#         super().__init__(func, *args, **kwargs)

#     def compute_output_shape(self, input_shape):
#         assert len(input_shape) >= 2
#         return input_shape
