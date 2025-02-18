from keras.models import Model
from keras.layers import Input, Conv3D, MaxPool3D, Conv3DTranspose, concatenate, Concatenate, BatchNormalization, Activation 

#convolution block
def conv_block(input, num_filters):
    s = Conv3D(num_filters, (3,3,3), padding="same")(input)
    s = BatchNormalization()(s)
    s = Activation("relu")(s)
    print("Conv1: ",s.shape)
    
    s = Conv3D(num_filters, (3,3,3), padding="same")(s)
    s = BatchNormalization()(s)
    s = Activation("relu")(s)
    print("Conv2: ",s.shape)
    return s

#encoder block
def encoder_block(input, num_filters):
    print("Encoder1: ",input.shape)
    s = conv_block(input, num_filters)
    p = MaxPool3D((2,2,2))(s)
    print("MaxPooling: ", p.shape)
    return s, p

#decoder block
def decoder_block(input, skip_features, num_filters):
    print("Decoder: ",input.shape)
    print("Skip_features Shape", skip_features.shape)
    s = Conv3DTranspose(num_filters, 2, strides=2, padding="same")(input)
    print("Concatenation: ",s.shape)
    s = Concatenate()([s, skip_features])
    s = conv_block(s, num_filters)
    
    return s

#model building
def build_model(input_shape):
   
    input = Input(input_shape)
    print("input: ",input.shape)
    s1, p1 = encoder_block(input, 128)
    s2, p2 = encoder_block(p1, 256)
    
    b1 = conv_block(p2, 512)
    
    d1 = decoder_block(b1, s2, 256)
    d2 = decoder_block(d1, s1, 128)
    
    output = Conv3D(1, 1, padding="same", activation="sigmoid")(d2)
    model = Model(input, output, name = "U-net")
    
    return model
    
    
    
    