import tensorflow as tf

from tensorflow.keras.layers import Conv2D,  BatchNormalization, Activation
from tensorflow.keras.layers import Concatenate, AveragePooling2D,Conv2DTranspose


def Inception(inputs):

    branch1 = Conv2D(32,1,padding="same")(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation("relu")(branch1)


    branch2 = Conv2D(32,1,padding="same")(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation("relu")(branch2)

    branch2 = Conv2D(32,kernel_size=(1,3),padding="same")(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation("relu")(branch2)

    branch2 = Conv2D(32,kernel_size=(3,1),padding="same")(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation("relu")(branch2)


    branch3 = Conv2D(32, 1, padding="same")(inputs)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation("relu")(branch3)

    branch3 = Conv2D(32, kernel_size=(1, 3), padding="same")(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation("relu")(branch3)

    branch3 = Conv2D(32, kernel_size=(3, 1), padding="same")(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Activation("relu")(branch3)


    x = Concatenate(axis=3)([branch1,branch2,branch3])
    return x

def Downsampling_block(Down,filters):

    down1 = AveragePooling2D(2,strides=2)(Down)

    down2 = Conv2D(filters,1,padding="same")(Down)
    down2 = BatchNormalization()(down2)
    down2 = Activation("relu")(down2)

    down2 = Conv2D(filters,3,strides=2,padding="same")(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation("relu")(down2)

    down3 = Conv2D(filters, 1, padding="same")(Down)
    down3 = BatchNormalization()(down3)
    down3 = Activation("relu")(down3)

    down3 = Conv2D(filters, 3,strides=2, padding="same")(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation("relu")(down3)



    down =  Concatenate(axis=3)([down1,down2,down3])
    return down




def dense_block(dense):
    x1 = Inception(dense)
    concat1 = Concatenate(axis=3)([dense,x1])
    x2 = Inception(concat1)
    concat2 = Concatenate(axis=3)([dense,x1,x2])
    x3 = Inception(concat2)
    concat3 = Concatenate(axis=3)([dense,x1,x2,x3])
    return concat3

def create_model():
    inputs = tf.keras.layers.Input(shape=(256,256,3))

    conv1 = Inception(inputs)

    dense_block1 = dense_block(conv1)
    down1 = Downsampling_block(dense_block1,filters=256)

    dense_block2 = dense_block(down1)
    down2 = Downsampling_block(dense_block2,filters=256)

    dense_block3 = dense_block(down2)
    down3 = Downsampling_block(dense_block3,filters=256)

    dense_block4 = dense_block(down3)
    down4 = Downsampling_block(dense_block4,filters=256)

    dense_block5 = dense_block(down4)

    up1 = Conv2DTranspose(512,3,strides=2,activation="relu",padding="same")(dense_block5)
    up_concat1 = Concatenate(axis=3)([up1,dense_block4])
    up_b1 = Inception(up_concat1)
    up_dense1 = dense_block(up_b1)


    up2 = Conv2DTranspose(512,3,strides=2,activation="relu",padding="same")(up_dense1)
    up_concat2 = Concatenate(axis=3)([up2,dense_block3])
    up_b2 = Inception(up_concat2)
    up_dense2 = dense_block(up_b2)

    up3 = Conv2DTranspose(512,3,strides=2,activation="relu",padding="same")(up_dense2)
    up_concat3 = Concatenate(axis=3)([up3,dense_block2])
    up_b3 = Inception(up_concat3)
    up_dense3 = dense_block(up_b3)


    up4 = Conv2DTranspose(512,3,strides=2,activation="relu",padding="same")(up_dense3)
    up_concat4 = Concatenate(axis=3)([up4,dense_block1])
    up_b4 = Inception(up_concat4)
    up_dense4 = dense_block(up_b4)

    prediction = tf.keras.layers.Convolution2D(2, 1, padding="same", activation="softmax")(up_dense4)  # 256*256*34
    return tf.keras.Model(inputs=inputs, outputs=prediction)

model = create_model()


