# Imports
import logging as log
import json
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, TimeDistributed
from keras_nlp.layers import TransformerEncoder, PositionEmbedding
from keras.optimizers import Adam
from keras.initializers import RandomUniform

# Constants
DATA_DIR = '../data/processed/checkpoint/'
MODEL_DIR = '../data/models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json'

NUM_CLASSES = 3

WINDOW_SIZE = 150
INPUT_DIMS = 2
EMBED_DIM = 512

NUM_HEADS = 4
NUM_LAYERS = 2
D_MODEL = 64
DFF = 256

N_FILTERS = 64
POOL_SIZE =2
FILTER_SIZE = 3

NUM_EPOCHS = 10
BATCH_SIZE = 64*8
DROPOUT_RATE = 0.1


# Logging
log.basicConfig(level=log.DEBUG,
                format='%(asctime)s %(levelname)s %(message)s', filename='../logs/training.bs32.embDim512.nlay2.dr01.dmdl64.nh4.nepch100.log')

# Set GPUs
def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        log.error()
        raise ValueError('No GPUs found!')
    log.info(f'GPUs found: {gpus}')
    print("Num GPUs Available: ", len(gpus))


# Data loading
def load_series(series_id):
    try:
        return np.load(os.path.join(DATA_DIR, series_id + '.npy'))
    except Exception as e:
        log.error(f'Error loading series {series_id}: {e}')

def load_batched_data(dataset_id):
    with open(DATASET_ALLOCATIONS, 'r') as f:
        allocations = json.load(f)
        series_ids = allocations[dataset_id]
        for series_id in series_ids:
            data = np.load(os.path.join(
                DATA_DIR, series_id + '.npz'), allow_pickle=True)
            x, y, events = data['sequences'], data['labels'], data['events']

            log.info(f'Loading data for series {series_id} with {len(x)} observations')
            # TESTING 
            yield x, y, events
            ##############################
            #for i in range(0, len(x), BATCH_SIZE):
            #    yield x[i:i+BATCH_SIZE], y[i:i+BATCH_SIZE], events[i:i+BATCH_SIZE]

def one_hot_encode_labels(labels):
    # Convert labels to one-hot encoding
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=3)
    return one_hot_labels


# Transformer components
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
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

# Metrics and Callbacks
def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 3 == 0:
        return lr * 0.9
    return lr

auc = tf.keras.metrics.AUC(multi_label=True, num_labels=3,
                           label_weights=[.0001, .4995, .4995], name='my_AUC')

def custom_auc(y_true, y_pred):
    # Flatten the last two dimensions
    y_true_reshaped = tf.reshape(y_true, [-1, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 3])

    return auc(y_true_reshaped, y_pred_reshaped)


# Model definition
def create_transformer_model(input_shape=(WINDOW_SIZE, INPUT_DIMS), training=True):
    inputs = Input(shape=input_shape)

    x = keras.layers.Conv1D(filters=N_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')(inputs)
    x = keras.layers.Conv1D(filters=N_FILTERS*2, kernel_size=FILTER_SIZE, activation='relu', padding='same')(x)

    positional_embedding_layer = PositionEmbedding(
        input_dim=N_FILTERS*2, sequence_length=WINDOW_SIZE)
    
    x += positional_embedding_layer(x)

    for _ in range(NUM_LAYERS):
        x = TransformerBlock(N_FILTERS*2, NUM_HEADS, DFF,
                             DROPOUT_RATE if training else 0)(x)
        
    outputs = keras.layers.TimeDistributed(Dense(NUM_CLASSES, activation='softmax'))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy', metrics=[custom_auc])
    return model


# Training Function
def train_model(model, dataset_id, limit_train=False):
    for i, (batch_x, batch_y, _) in enumerate(load_batched_data(dataset_id)):
        log.info(f"Batch {i}: X Shape {batch_x.shape}, Y Shape {batch_y.shape}")
        batch_y = one_hot_encode_labels(batch_y)
        if not limit_train or (limit_train and i < 1000):
            history = model.fit(batch_x, batch_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, verbose=1)
            log.info(
                    f'Training finished. Loss: {history.history["loss"][-1]}, Accuracy: {history.history["custom_auc"][-1]}')

def main():
    try: 
        setup_gpus()
    except Exception as e:
        log.error(f'Error setting up GPUs: {e}')
        raise e
    log.info('GPUs set up.\nDefining model...')

    try:
        model = create_transformer_model()
        # Log the model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        log.info("\n".join(model_summary))
    except Exception as e:
        log.error(f'Error defining model: {e}')
        raise e
        
    log.info(f'Model defined.\n{model.summary()})')
    

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    train_model(model, 'train', limit_train=False)

    # Save the model
    model.save(os.path.join(MODEL_DIR, 'ConvFormer.v1.h5'))

# Main
if __name__ == '__main__':
    main()