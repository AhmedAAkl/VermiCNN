from model import DenseNetLoader
from preprocess import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import random


class Predictor:


    def __init__(self) -> None:
        pass

    

    def predict(self, model, image):

        densenet_preds = model.predict(image)
        densenet_preds = densenet_preds.argmax(axis=-1)

        return densenet_preds

    



    def evaluate_model(self, model, x_train, y_train, x_test, y_test):

        densenet_train_acc = model.evaluate(x_train, y_train)
        model.evaluate(x_test, y_test)

        densenet_preds = model.predict(x_test)
        densenet_preds = densenet_preds.argmax(axis=-1)
        densenet_f1_score = f1_score(y_test, densenet_preds, average='micro')
        densenet_test_acc = accuracy_score(y_test, densenet_preds)
        print("Densenet Model Train Acc.: ", densenet_train_acc)
        print("Densenet Model Test Acc.: ", densenet_test_acc)
        print("Densenet Model F1 Score: ", densenet_f1_score)
        print(classification_report(y_test, densenet_preds))









if __name__ == '__main__':


    IMAGE_SIZE = 224
    NUM_CLASSES = 3
    BATCH_SIZE = 32
    EPOCHS = 100


    
    output_dir = "src/model_weights/"

    # load the training and testing datasets
    data_dir = 'data/'
    data_path = data_dir + 'data.csv'
    dataloader = DataLoader(data_path)

    x_train, x_test, y_train, y_test = dataloader.preprocess_data(data_dir, IMAGE_SIZE)
    print("Data loaded correctly.")

    densenet_loader = DenseNetLoader(IMAGE_SIZE, NUM_CLASSES)        
    densenet_model = densenet_loader.get_DenseNet121_model()
    print("DenseNet model created correclty.")

    densenet_model.load_weights(output_dir + 'densenet_121_model.hdf5')


    predictor = Predictor()

    print(x_train[:1].shape)

    # select random image index from the test dataset.
    rand_id = random.sample(range(0, x_test.shape[1]-1), 1)
    rand_test_img = x_test[rand_id, :]
    
    

    img_label = predictor.predict(densenet_model, rand_test_img)

    print("Image True Label: ", y_test[rand_id], " Predicted Label: " ,img_label)

    # uncomment the to display the model's performance on both the training and testing datasets. 
    print("Model Results on Test Data: ")
    predictor.evaluate_model(densenet_model, x_train, y_train, x_test, y_test)

