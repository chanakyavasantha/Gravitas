import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD
from tensorflow import convert_to_tensor
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_(image_path):
    # Load the Model from Json File
    json_file = open('model.json', 'r')
    model_json_c = json_file.read()
    json_file.close()
    model_c = model_from_json(model_json_c)

    # Load the weights
    model_c.load_weights("best_model.h5")

    # Compile the model
    opt = SGD(lr=1e-4, momentum=0.9)
    model_c.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))  # Adjust target size as needed
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the image
    preds = model_c.predict(convert_to_tensor(image))[0]
    print(preds)
    
    # Get the predicted class index with the highest probability
    predicted_label = np.argmax(preds)

    classes = {
        0: "Coast",
        1: "Desert",
        2: "Forest",
        3: "Glacier",
        4: "Mountain",
    }

    print("Predicted Label:", classes[predicted_label])
    return classes[predicted_label]

# Call the predict_ function with the image path
