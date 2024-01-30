import argparse
import logging as log
import os
import sys
import tensorflow as tf

# Add custom directory for functions
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
src_dir = os.path.normpath(src_dir)
sys.path.append(src_dir)
from data_loader import DataLoader
from misc import setup_gpus, load_model

# Constants
LOG_DIR = '../logs/'
DATA_DIR = '../data/processed/checkpoint/'
MODEL_DIR = '../models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json'

NUM_EPOCHS = 10
BATCH_SIZE = 32 * 16

# Argument Parsing
parser = argparse.ArgumentParser(description='Train a specified model.')
parser.add_argument('--model_name', type=str, default='convolution_transformer_v00', help='Name of the model to train')
args = parser.parse_args()

# Logging Setup
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                filename=os.path.join(LOG_DIR, f'training.{args.model_name}.log'))
logger = log.getLogger()

# Training Functions
class EndEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(f"Epoch {epoch + 1}: Loss: {logs.get('loss')}, Custom AUC: {logs.get('custom_auc')}")

def train_model(model, dataset_id, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    custom_logger = EndEpochLogger()
    data_loader = DataLoader(dataset_id=dataset_id, batch_size=batch_size)
    history = model.fit(data_loader, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[custom_logger])
    log.info(f'Training finished. Loss: {history.history["loss"][-1]}, Custom AUC: {history.history["custom_auc"][-1]}')

# Main Function
def main(model_name):
    setup_gpus(logger)
    model = load_model(MODEL_DIR, model_name, logger)
    
    # Log the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    log.info("\n".join(model_summary))
    
    train_model(model, 'train', NUM_EPOCHS, BATCH_SIZE)
    log.info('Model trained')
    
    model.save(os.path.join(MODEL_DIR, model_name))
    log.info('Model Saved.\nDone.')

# Execute Main Function
if __name__ == '__main__':
    main(args.model_name)
