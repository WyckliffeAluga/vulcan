from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

datagen = ImageDataGenerator(rescale=1.0/255.0)
train_it = datagen.flow_from_directory('formated_data/train/', class_mode='categorical', batch_size=32)
val_it = datagen.flow_from_directory('formated_data/valid/', class_mode='categorical', batch_size=32)
test_it = datagen.flow_from_directory('formated_data/test/', class_mode='categorical', batch_size=32)


model = Sequential()

# 1st convolutional network
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256, 256 , 3)))

# 2nd convolutional network
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 3rd convolutional network
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 4th convolutional network
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 5th convolutional network
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# 6th convolutional network
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(38, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit_generator(train_it, 
                    steps_per_epoch=32,
                    epochs=100, 
                    validation_data=val_it, 
                    validation_steps=8)

model.save("model.h5")


