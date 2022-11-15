from operator import mod
from statistics import mode
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers
from tensorflow.keras import models




class DenseNetLoader:

    def __init__(self, IMAGE_SIZE, NUM_CLASSES) -> None:
        self.IMAGE_SIZE = IMAGE_SIZE
        self.NUM_CLASSES = NUM_CLASSES


    def get_DenseNet121_model(self):

        base_model = DenseNet121(include_top=False, input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3))
        base_model.trainable = False
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dense(128, activation='relu')(x)
        x= layers.Dropout(0.25)(x)
        output = layers.Dense(self.NUM_CLASSES, activation='softmax')(x)
        densenet_model = models.Model(inputs=[base_model.input], outputs=[output])

        densenet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return densenet_model

    def model_summary(self):
        model = self.get_DenseNet121_model()
        print("Model Summary: ")
        print(model.summary())







if __name__ == '__main__':

    NUM_CLASSES = 3
    IMAGE_SIZE = 224

    densenet_loader = DenseNetLoader(IMAGE_SIZE, NUM_CLASSES)

    dense_model = densenet_loader.get_DenseNet121_model()


    densenet_loader.model_summary()