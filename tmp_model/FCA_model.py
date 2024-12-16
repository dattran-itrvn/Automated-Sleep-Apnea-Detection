import tensorflow as tf
import keras

def dct_2d(
        feature_map,
        norm=None
):
    X1 = tf.signal.dct(feature_map, type=2, norm=norm)
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.dct(X1_t, type=2, norm=norm)
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    return X2_t

class FCA_Block(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FCA_Block, self).__init__(**kwargs)
    
    def build(self, input_shape):
        _, _, _, C = input_shape # get channel
        self.dense1 = keras.layers.Dense(C, activation='relu')
        self.dense2 = keras.layers.Dense(C, activation='sigmoid')

    def call(self, inputs, training=False):
        _, _, _, C = inputs.shape
        # Decompose the input feature map into frequency channels using 2D DCT
        split_x = tf.split(inputs, C, axis=-1)
        dct_x = [dct_2d(channel) for channel in split_x]

        # Pass each channel through a fully connected layer to obtain the scalar weights
        weights = self.dense2(self.dense1(tf.concat(dct_x, axis=-1)))
        
        # Multiply the weights with the corresponding frequency channels
        weighted_x = [split_x[i] * weights[:, :, :, i:i+1] for i in range(C)]
        weighted_x = tf.concat(weighted_x, axis=-1)
        
        return weighted_x

    def get_config(self):
        config = super(FCA_Block, self).get_config().copy()
        config.update({
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ConvBlock(keras.layers.Layer):
    def __init__(self, out_filters, kernel_size,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.out_filters = out_filters
        self.kernel_size = kernel_size
        
        self.conv1 = keras.layers.Conv2D(out_filters, kernel_size, padding='same')
        self.conv2 = keras.layers.Conv2D(out_filters, kernel_size, padding='same')
        
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        
        self.activation1 = keras.layers.LeakyReLU()
        self.activation2 = keras.layers.LeakyReLU()
        
        self.add1 = keras.layers.Add()
        self.add2 = keras.layers.Add()
        
        self.fca = FCA_Block()
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        tmp_x = self.bn1(x, training=training)
        
        x = self.fca(x, training=training)
        
        x = self.add1([x, tmp_x])
        
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation2(x)
        
        x = self.add2([x, inputs])
        return x

    def get_config(self):
        config = super(ConvBlock, self).get_config().copy()
        config.update({
            'out_filters': self.out_filters,
            'kernel_size': self.kernel_size,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ResBlock(keras.layers.Layer):
    def __init__(self, conv_filters=[64, 64, 128, 256, 512], kernel_sizes=[7, 3, 3, 3, 3], 
                 first_pooling=2, 
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        
        self.conv_filters = conv_filters 
        self.kernel_sizes = kernel_sizes 
        self.first_pooling = first_pooling
        
        self.convblock1 = ConvBlock(out_filters=conv_filters[0], kernel_size=kernel_sizes[0])
        self.pooling = keras.layers.MaxPool2D(pool_size=first_pooling)
        
        self.convblock2 = keras.Sequential([ConvBlock(out_filters=conv_filters[1], kernel_size=kernel_sizes[1]) for _ in range(2)]) # x2
        self.convblock3 = keras.Sequential([ConvBlock(out_filters=conv_filters[2], kernel_size=kernel_sizes[2]) for _ in range(2)]) # x2
        self.convblock4 = keras.Sequential([ConvBlock(out_filters=conv_filters[3], kernel_size=kernel_sizes[3]) for _ in range(3)]) # x3
        self.convblock5 = keras.Sequential([ConvBlock(out_filters=conv_filters[4], kernel_size=kernel_sizes[4]) for _ in range(3)]) # x3

        
    def call(self, inputs, training=False):
        x = self.convblock1(inputs)
        x = self.pooling(x)
        
        x = self.convblock2(x)
        
        x = self.convblock3(x)
        
        x = self.convblock4(x)
        
        x = self.convblock5(x)
        
        return x

    def get_config(self):
        config = super(ResBlock, self).get_config().copy()
        config.update({
            'conv_filters': self.conv_filters,
            'kernel_sizes': self.kernel_sizes,
            'first_pooling': self.first_pooling,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Proposed_Model(keras.Model):
    def __init__(self, conv_filters=[64, 64, 128, 256, 512], kernel_sizes=[7, 3, 3, 3, 3], first_pooling=2,
                 fc_units=[512, 64, 16], dropout_rate=0.5,
                  
                 **kwargs):
        super(Proposed_Model, self).__init__(**kwargs)
        
        self.conv_filters = conv_filters 
        self.kernel_sizes = kernel_sizes 
        self.first_pooling = first_pooling
        
        self.resblock = ResBlock(conv_filters, kernel_sizes, first_pooling)
        
        self.fca = FCA_Block()
        
        self.add = keras.layers.Add()
        
        self.gap = keras.layers.GlobalAveragePooling2D() # I assume using GAP (paper just mention avg pool with no size)
        
        self.dense1 = keras.layers.Dense(fc_units[0])
        self.dense2 = keras.layers.Dense(fc_units[1])
        
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        
        self.classify = keras.layers.Dense(fc_units[2], activation='softmax')
        
    def call(self, inputs, training=False):
        x = self.resblock(inputs, training=training)
        tmp_x = self.fca(x, training=training)
        x = self.add([x, tmp_x])
        
        x = self.gap(x)
        
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        x = self.classify(x)
        return x

    def get_config(self):
        config = super(Proposed_Model, self).get_config().copy()
        config.update({
            'conv_filters': self.conv_filters,
            'kernel_sizes': self.kernel_sizes,
            'first_pooling': self.first_pooling,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Example usage
if __name__ == "__main__":
    # Dummy input: Batch size 4, Height 32, Width 32, Channels 3
    input_tensor = tf.random.normal([4, 32, 32, 3])

    model = Proposed_Model()
    
    output_tensor = model(input_tensor)
    
    print("Input Shape:", input_tensor.shape)
    print("Output Shape:", output_tensor.shape)
