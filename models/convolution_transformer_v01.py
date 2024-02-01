import os, sys
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, 
                                     BatchNormalization, Conv1D, TimeDistributed, Conv1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform
from keras_nlp.layers import PositionEmbedding
import keras.callbacks

# Add custom directory for functions
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
src_dir = os.path.normpath(src_dir)
sys.path.append(src_dir)

from misc import custom_auc

# Constants
MODEL_NAME = 'convolution_transformer_v01'

NUM_CLASSES = 3
WINDOW_SIZE = 150
INPUT_DIMS = 2
NUM_HEADS = 4
NUM_LAYERS = 2
D_MODEL = 64
DFF = 256
N_FILTERS = 64
FILTER_SIZE = 3
DROPOUT_RATE = 0.5
LEARNING_RATE = 1e-5


# Model Definition
class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block class.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dropout(rate),
            Dense(embed_dim, kernel_initializer=RandomUniform(
                minval=-0.01, maxval=0.01))
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

def create_transformer_model(input_shape=(WINDOW_SIZE, INPUT_DIMS), training=True):
    """
    Create and compile the transformer model.
    """    
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=N_FILTERS*.5, kernel_size=FILTER_SIZE, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x) 
    x = Conv1D(filters=N_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')(x) # 2.2 REMOVED THIS LAYER
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv1D(filters=N_FILTERS*2, kernel_size=FILTER_SIZE, activation='relu', padding='same')(x) # 2.2 REMOVED THIS LAYER
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)
    positional_embedding_layer = PositionEmbedding(
        input_dim=N_FILTERS*2, sequence_length=WINDOW_SIZE)
    x += positional_embedding_layer(x)
    for _ in range(NUM_LAYERS):
        x = TransformerBlock(N_FILTERS*2, NUM_HEADS, DFF,
                             DROPOUT_RATE if training else 0)(x)
    outputs = TimeDistributed(Dense(NUM_CLASSES, activation='softmax'))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=[custom_auc])
    return model

def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function.
    """
    if epoch > 0 and epoch % 2 == 0:
        return lr * 0.9
    return lr

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_custom_auc', patience=3, mode='max', restore_best_weights=True)
lr_schedule = keras.callbacks.LearningRateScheduler(lr_scheduler)

# Model Saving
def save_model(model, model_dir, model_name):
    """
    Save the model to the specified directory.
    """
    try:
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
    except Exception as e:
        print(f"Error saving the model: {e}")


# Main
if __name__ == "__main__":
    model = create_transformer_model()
    save_model(model, os.getcwd(), MODEL_NAME)
    print("Model saved.")
