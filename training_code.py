import numpy as np
import tensorflow as tf

'''  defining the training setand validation set from a folder on google colab  
    
'''
train_set = tf.keras.utils.image_dataset_from_directory, 
directory = ("./output_folder/Final images/train"),
labels = "inferred",
label_mode = "categorical",
color_mode = "rgb",
batch_size = 8,
image_size = (512,512),
shuffle = True,
seed = 42)


validation_set = tf.keras.utils.image_dataset_from_directory(
directory = ("./output_folder/Final images/val"),
labels = "inferred",
label_mode = "categorical",
color_mode = "rgb",
batch_size = 8,
image_size = (512,512),
shuffle = True,
seed = 42)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(512,512, 3),
    alpha=1.0,
    include_top = False,
    weights="imagenet",
    )

base_model.trainable = False 

model = tf.keras.Sequential([
    tf.keras.Input(shape=(512, 512, 3)), 
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x)), 
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(18, activation="softmax"),
])
autotune = tf.data.AUTOTUNE
train_set = train_set.prefetch(autotune)
validation_set = validation_set.prefetch(buffer_size= autotune)

model_checkpoint_callbacks=tf.keras.callbacks.ModelCheckpoint(
    filepath = "./model.{epoch:02d}-{val_accuracy:.2f}.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
)
model.compile(
    optimizer="adam",
    loss=tf.losses.CategoricalCrossentropy(),
    metrics=["accuracy"]
)


history = model.fit(
    train_set,
    validation_data = validation_set,
    epochs= 50,
    callbacks=[model_checkpoint_callbacks]
)
