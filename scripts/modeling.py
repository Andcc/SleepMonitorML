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
from keras.regularizers import l1_l2


# Constants
MODEL_NAME = 'ConvFormer.v3' # v2:add Dropout layers ; 2.2 add l1_l2, LR sched, and early stop. ; 3.0 revert to 1:0
LOG_NAME = f'../logs/training.{MODEL_NAME}.log'

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
FILTER_SIZE = 3

NUM_EPOCHS = 10 #1.0:10 ; 1.1:15 ; 2.0:15 ; 
BATCH_SIZE = 32*16 #1.0:512 ; 1.1:128 ; 2.0:128 ; 2::512 ;
DROPOUT_RATE = 0.1 #1.0:0.1 ; 1.1:0.5 ; 2.0:0.5 ; 

AUC = tf.keras.metrics.AUC(multi_label=True, num_labels=3,
                           label_weights=[.0001, .4995, .4995], name='my_AUC')

# Logging
log.basicConfig(level=log.DEBUG,
                format='%(asctime)s %(levelname)s %(message)s', filename=LOG_NAME)

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
        
def augment_data(x, noise_level=0.01):
    noise = np.random.normal(0, noise_level, x.shape)
    return x + noise

def load_all_data(dataset_id):
    with open(DATASET_ALLOCATIONS, 'r') as f:
        allocations = json.load(f)
        series_ids = allocations[dataset_id]

        all_x, all_y, all_events = [], [], []
        for series_id in series_ids:
            data = np.load(os.path.join(DATA_DIR, series_id + '.npz'), allow_pickle=True)
            x, y, events = data['sequences'], data['labels'], data['events']
            x = augment_data(x)
            all_x.append(x)
            all_y.append(y)
            all_events.append(events)

            log.info(f'Loading data for series {series_id} with {len(x)} observations')

        # Concatenate all data
        all_x = np.concatenate(all_x, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        all_events = np.concatenate(all_events, axis=0)

        return all_x, all_y, all_events

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

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_custom_auc', patience=3, mode='max', restore_best_weights=True)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

def custom_auc(y_true, y_pred):
    # Flatten the last two dimensions
    y_true_reshaped = tf.reshape(y_true, [-1, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 3])
    return AUC(y_true_reshaped, y_pred_reshaped)


# Model definition
def create_transformer_model(input_shape=(WINDOW_SIZE, INPUT_DIMS), training=True):
    inputs = Input(shape=input_shape)

    x = keras.layers.Conv1D(filters=N_FILTERS, kernel_size=FILTER_SIZE, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv1D(filters=N_FILTERS*2, kernel_size=FILTER_SIZE, activation='relu', padding='same')(x) # 2.2 REMOVED THIS LAYER
    x = keras.layers.BatchNormalization()(x)

    positional_embedding_layer = PositionEmbedding(
        input_dim=N_FILTERS*2, sequence_length=WINDOW_SIZE)
    
    x += positional_embedding_layer(x)

    for _ in range(NUM_LAYERS):
        x = TransformerBlock(N_FILTERS*2, NUM_HEADS, DFF,
                             DROPOUT_RATE if training else 0)(x)
        
    outputs = keras.layers.TimeDistributed(Dense(NUM_CLASSES, activation='softmax'))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=5e-5),
                  loss='categorical_crossentropy', metrics=[custom_auc])
    return model


# Training Function
class CustomLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(f"Epoch {epoch + 1}: Loss: {logs.get('loss')}, Custom AUC: {logs.get('custom_auc')}")

def train_model(model, dataset_id, batch_size=BATCH_SIZE):
    all_x, all_y, _ = load_all_data(dataset_id)
    all_y = one_hot_encode_labels(all_y)
    log.info(f"X Shape {all_x.shape}, Y Shape {all_y.shape}")

    custom_logger = CustomLogger()
    history = model.fit(all_x, all_y, batch_size=batch_size, epochs=NUM_EPOCHS, verbose=1, callbacks=[custom_logger])
    log.info(f'Training finished. Loss: {history.history["loss"][-1]}, Custom AUC: {history.history["custom_auc"][-1]}')



# Main
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
        
    log.info(f'Model defined.\n{model.summary()}')
    train_model(model, 'train', BATCH_SIZE)
    log.info('Model trained')
    # Save the model
    model.save(os.path.join(MODEL_DIR, MODEL_NAME))
    log.info('Done.')

# Main
if __name__ == '__main__':
    main()