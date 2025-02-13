from keras.models import Model
from keras.layers import Input, Conv3D, MaxPool3D, Conv3DTranspose, concatenate, Concatenate, BatchNormalization, Activation 

#convolution block
def conv_block(input, num_filters):
    s = Conv3D(num_filters, 3, padding="same")(input)
    s = BatchNormalization()(s)
    s = Activation("relu")(s)
    
    s = Conv3D(num_filters, 3, padding="same")(s)
    s = BatchNormalization()(s)
    s = Activation("relu")(s)
    
    return s

#encoder block
def encoder_block(input, num_filters):
    s = conv_block(input, num_filters)
    p = MaxPool3D((2,2,2))(s)
    
    return s, p

#decoder block
def decoder_block(input, skip_features, num_filters):
    s = Conv3DTranspose(num_filters, (3,3), strides=3, padding="same")(input)
    s = Concatenate()([s, skip_features])
    s = conv_block(s, num_filters)
    
    return s

#model building
def build_model(input_shape):
   
    input = Input(input_shape)
   
    s1, p1 = encoder_block(input, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    b1 = conv_block(p4, 1024)
    
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    output = Conv3D(1, 1, padding="same", activation="sigmoid")(d4)
    model = Model(input, output, name = "U-net")
    
    return model
    
    
    
    