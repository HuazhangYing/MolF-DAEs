import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

class Encoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1024, activation='relu')
        self.d2 = Dense(512, activation='relu')
        self.d3 = Dense(128, activation='relu')
        self.d4 = Dense(64, activation='relu')
        self.d5 = Dense(32, activation='relu')
        self.d6 = Dense(3, activation='relu')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        return self.d6(x)       

class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d7 = Dense(32, activation='relu')
        self.d8 = Dense(64, activation='relu')
        self.d9 = Dense(128, activation='relu')
        self.d10 = Dense(512, activation='relu')
        self.d11 = Dense(1024, activation='relu')
        self.d12 = Dense(169, activation='sigmoid')
        self.re = Reshape((13,13))
    
    def call(self, x):
        x = self.d7(x)
        x = self.d8(x)
        x = self.d9(x)
        x = self.d10(x)
        x = self.d11(x)
        x = self.d12(x)
        return self.re(x)
        
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def create_and_train_model(X, batch_size=768, epochs=100):
    model = Autoencoder()
    
    # 第一阶段训练
    model.compile(optimizer='Adam', loss='binary_crossentropy')
    history1 = model.fit(X, X, batch_size=batch_size, epochs=epochs)
    
    # 第二阶段训练
    model.compile(optimizer='Adam', loss='mse')
    history2 = model.fit(X, X, batch_size=batch_size, epochs=epochs)
    
    return model, history1, history2

def encode_data(model, X):
    return model.encoder(X)