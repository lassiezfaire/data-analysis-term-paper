from keras import layers
from keras import models

class Model(models.Sequential):
    def __init__(self):
        super(Model, self).__init__()
        self.add(layers.Flatten(input_shape=(28, 28)))
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dense(10, activation='softmax'))
        self.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])