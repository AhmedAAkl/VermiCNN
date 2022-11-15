from model import DenseNetLoader
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from preprocess import DataLoader
import numpy as np

class Trainer:

    def __init__(self) -> None:
         pass
    


    def finetune_DenseNet(self, model, x_train, y_train, EPOCHS, BATCH_SIZE, model_callbacks):        
  
        try:
            model_history = model.fit(x_train, y_train, callbacks=model_callbacks, batch_size=BATCH_SIZE, 
                                    epochs=EPOCHS, verbose=1,validation_split=0.1)

        except KeyboardInterrupt:
                print("Execution interrputed by user.")



if __name__ == '__main__':

    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 100


    
    output_dir = "src/model_weights/"


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.000001)
    early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
    checkpoint_filepath = output_dir + 'class_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    densenet_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model_callbacks = [reduce_lr, early_stop, densenet_checkpoint_callback]


    # load the training and testing datasets
    data_dir = 'data/'
    data_path = data_dir + 'data.csv'
    dataloader = DataLoader(data_path)

    x_train, x_test, y_train, y_test = dataloader.preprocess_data(data_dir, IMAGE_SIZE)
    print("Data loaded correctly.")
    
    
    densenet_loader = DenseNetLoader(IMAGE_SIZE, NUM_CLASSES)        
    densenet_model = densenet_loader.get_DenseNet121_model()
    print("DenseNet model created correclty.")


    trainer = Trainer()
    print("Start the training process...")
    trainer.finetune_DenseNet(densenet_model, x_train, y_train, EPOCHS, BATCH_SIZE, model_callbacks)