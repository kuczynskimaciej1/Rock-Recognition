import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import pathlib

def initial():
    # Set up paths
    #dataset_url = r"C:\Users\mkuczyns\Downloads\RockRecognition\Dataset"
    dataset_url = r"C:\Users\mkuczyns\Downloads\RockRecognition\Rocks"
    dataset_dir = pathlib.Path(dataset_url)
    test_url = r"C:\Users\mkuczyns\Downloads\RockRecognition\Testset"
    test_dir = pathlib.Path(test_url)

    print("1. Trenuj")
    print("2. Testuj")
    mode = 0

    while(mode != {1, 2}):
        mode = input("Opcja: ")

        if (mode == "1"):
            train(dataset_dir, test_dir)
        elif (mode == "2"):
            test()
        else:
            print("Zly wybor")
    return

def train(dataset_dir, test_dir):
    # Define hyperparameters
    batch_size = 32
    num_epochs = 40
    input_shape = (256, 256, 3) #poprzednio wylaczone
    num_classes = 29

    # Data augmentation and preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=(input_shape[0], input_shape[1]), #poprzednio wylaczone
        color_mode="rgb", #nowe
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(input_shape[0], input_shape[1]), #poprzednio wy��czone
        color_mode="rgb", #nowe
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu')) #nowe
    model.add(MaxPooling2D((2, 2)))
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu')) #by�o 128
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint("rock_identification_model_Astronaut.h5", monitor='loss', verbose=1, save_weights_only=False, mode='auto', save_freq=1)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        validation_data=test_generator,
        validation_steps=len(test_generator),
        callbacks=[checkpoint]
    )

    # Save the model
    model.save('rock_identification_model_Astronaut.h5')
    print("Model saved successfully.")

def test():
    # Load the trained model
    model = tf.keras.models.load_model('rock_identification_model_Astronaut.h5')

    # Load and preprocess the test image
    chosen_img = input("Zdjecie do testowania (44, 55, 66): ")
    img_url = f"C:/Users/mkuczyns/Downloads/RockRecognition/Testset/{chosen_img}.PNG"
    img_dir = pathlib.Path(img_url)

    img = tf.keras.preprocessing.image.load_img(
        img_dir, target_size=(256, 256)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array /= 255.0

    for i in range (10):
        # Predict the class of the test image
        predictions = model.predict(img_array)
        predicted_class = predictions.argmax(axis=1)[0]

        # Define class labels
        class_labels = [
        "Basalt",
        "Breccia",
        "Chert",
        "Conglomerate",
        "Dolomite",
        "Gabbro",
        "Gneiss",
        "Granite",
        "Granulite",
        "Greenschist",
        "Hornfels",
        "Limestone",
        "Marble",
        "Mudstone",
        "Obsidian",
        "Phyllite",
        "Porphyry",
        "Quartz_diorite",
        "Quartz_monzonite",
        "Quartzite",
        "Rhyolite",
        "Sandstone",
        "Scoria",
        "Serpentinite",
        "Shale",
        "Siltstone",
        "Slate",
        "Tuff"
        ]

        # Print the predicted class label
        print(f"Image: {chosen_img}")
        print("Predicted class:", class_labels[predicted_class])

initial()