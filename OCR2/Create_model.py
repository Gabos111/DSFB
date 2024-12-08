from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    GRU,
    TimeDistributed,
    Reshape,
    Activation,
    Input,
    concatenate,
    SeparableConv2D
)
from tensorflow.keras.optimizers import Adam

def create_simple_lenet(input_shape=(64, 32, 3), num_classes=13):
    """
    Simple implementation of the original LeNet architecture for input size 64x32.
    """
    model = Sequential()

    # Layer 1: Convolution + Activation + Pooling
    model.add(Conv2D(6, kernel_size=(5, 5), strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Layer 2: Convolution + Activation + Pooling
    model.add(Conv2D(16, kernel_size=(5, 5), strides=1, padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(120))
    model.add(Activation('tanh'))

    # Fully Connected Layer 2
    model.add(Dense(84))
    model.add(Activation('tanh'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Classification layer

    return model

def create_augmented_lenet1(input_shape=(64, 32, 3), num_classes=13):
    model = Sequential()

    # Layer 1: Convolution + Activation + Pooling
    model.add(Conv2D(16, kernel_size=(5, 5), strides=1, padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.2))  # Regularization

    # Layer 2: Convolution + Activation + Pooling
    model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    model.add(Dropout(0.3))  # Regularization

    # Layer 3: Convolution + Activation
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    # Global Average Pooling
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layer 1
    model.add(Dense(120, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))  # Regularization

    # Fully Connected Layer 2
    model.add(Dense(84, kernel_initializer='he_normal'))
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Classification layer

    return model

def create_augmented_lenet2(input_shape=(64, 32, 3), num_classes=13):
    """
    Refined version of LeNet with modern techniques for better performance.
    """
    model = Sequential()

    # Layer 1: Convolution + BatchNorm + Activation + Pooling
    model.add(Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Layer 2: Convolution + BatchNorm + Activation + Pooling
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Layer 3: Convolution + BatchNorm + Activation + Pooling
    model.add(Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
    
    # Flatten
    model.add(Flatten())

    # Fully Connected Layer 1 + BatchNorm + Dropout
    model.add(Dense(256, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))  # Increased dropout for regularization

    # Fully Connected Layer 2 + BatchNorm + Dropout
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))  # Classification layer

    return model
def create_alexnet(input_shape=(64, 32, 3), num_classes=13):
    """
    AlexNet adapté pour les dimensions 64x32 et 13 classes.
    """
    model = Sequential()

    # Bloc 1 : Convolution + Activation + MaxPooling
    model.add(Conv2D(96, kernel_size=(11, 11), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Bloc 2 : Convolution + Activation + MaxPooling
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Bloc 3 : Convolution
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Bloc 4 : Convolution
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Bloc 5 : Convolution + MaxPooling
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Flatten : Conversion en vecteur
    model.add(Flatten())

    # Fully Connected 1 + Dropout
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully Connected 2 + Dropout
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully Connected 3 : Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

def create_adapted_alexnet(input_shape=(64, 32, 3), num_classes=13):
    """
    Adapted AlexNet architecture for 64x32 images and 13 output classes.
    """
    model = Sequential()

    # Block 1: Convolution + BatchNorm + Activation + MaxPooling
    model.add(Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Block 2: Convolution + BatchNorm + Activation + MaxPooling
    model.add(Conv2D(128, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Block 3: Convolution
    model.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Block 4: Convolution
    model.add(Conv2D(192, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Block 5: Convolution + MaxPooling
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    # Flatten: Convert to vector
    model.add(Flatten())

    # Fully Connected 1 + Dropout
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Fully Connected 2 + Dropout
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model

def cnn_rnn_model(input_shape=(64, 32, 3), num_classes=13):
    model = Sequential()

    # Convolutional Block
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Reshape for RNN
    model.add(Reshape((16, -1)))  # Flatten along one spatial dimension

    # Recurrent Block
    model.add(GRU(128, return_sequences=False))

    # Dense Layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def multi_scale_cnn(input_shape=(64, 32, 3), num_classes=13):
    input_layer = Input(shape=input_shape)

    # Branch 1
    x1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
    x1 = MaxPooling2D((2, 2))(x1)

    # Branch 2
    x2 = Conv2D(16, (5, 5), activation='relu', padding='same')(input_layer)
    x2 = MaxPooling2D((2, 2))(x2)

    # Branch 3
    x3 = Conv2D(16, (7, 7), activation='relu', padding='same')(input_layer)
    x3 = MaxPooling2D((2, 2))(x3)

    # Concatenate
    merged = concatenate([x1, x2, x3])

    # Fully Connected Layers
    flat = Flatten()(merged)
    dense = Dense(64, activation='relu')(flat)
    output = Dense(num_classes, activation='softmax')(dense)

    return Model(inputs=input_layer, outputs=output)

def depthwise_cnn(input_shape=(64, 32, 3), num_classes=13):
    model = Sequential()

    # Depthwise Separable Convolutional Layers
    model.add(SeparableConv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def create_custom_Lenet_model(input_shape=(64, 32, 3), num_classes=13):
    """
    Creates a model with the specified architecture based on the provided summary.

    Parameters:
        input_shape (tuple): Shape of the input data (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()

    # Layer 1: Convolution + Activation + Pooling
    model.add(Conv2D(6, kernel_size=(5, 5), strides=1, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Layer 2: Convolution + BatchNorm + Activation + Pooling
    model.add(Conv2D(16, kernel_size=(5, 5), strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(120))
    model.add(Activation('relu'))

    # Fully Connected Layer 2
    model.add(Dense(84))
    model.add(Activation('relu'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model