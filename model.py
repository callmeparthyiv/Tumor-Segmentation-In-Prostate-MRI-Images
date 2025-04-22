from keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, concatenate, Concatenate, BatchNormalization, Activation
from tensorflow.keras.regularizers import  l2

#convolution block
def conv_block(input, num_filters):
    s = Conv3D(num_filters, (5,5,5),activation='relu', kernel_regularizer=l2(0.001), padding="same")(input)
    s = Conv3D(num_filters, (5,5,5),activation='relu', kernel_regularizer=l2(0.001), padding="same")(s)
    return s

#encoder block
def encoder_block(input, num_filters):
    s = conv_block(input, num_filters)
    p = MaxPooling3D(pool_size=(2,2,2))(s)
    return s, p

#decoder block
def decoder_block(input, skip_features, num_filters):  
    s = UpSampling3D(size = (2,2,2))(input)
    s = Conv3D(num_filters, 2, padding='same', kernel_regularizer=l2(0.001), activation='relu')(s)
    s = concatenate([s, skip_features], axis=-1)
    s = conv_block(s, num_filters)
    return s

#model building
def build_model(input_shape):
   
    input = Input(input_shape)
    s1, p1 = encoder_block(input, 32)
    s2, p2 = encoder_block(p1, 64)

    b1 = conv_block(p2, 128)

    d1 = decoder_block(b1, s2, 64)
    d2 = decoder_block(d1, s1, 32)

    output = Conv3D(1, 1, kernel_regularizer=l2(0.001), activation="sigmoid")(d2)
    model = Model(input, output, name = "U-net")
    
    return model
    
    
    
    