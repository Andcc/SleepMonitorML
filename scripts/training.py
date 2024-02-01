import logging as log
import os, sys, argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Add custom directory for functions
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
src_dir = os.path.normpath(src_dir)
sys.path.append(src_dir)
from data_loader import DataLoader
from misc import setup_gpus, load_model

# Argument Parsing
parser = argparse.ArgumentParser(description='Train a specified model.')
parser.add_argument('--model_name', type=str, default='convolution_transformer_v00', help='Name of the model to train')
parser.add_argument('--batch_size', type=int, default=512, help='Number of samples per batch')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
args = parser.parse_args()

# Constants
LOG_DIR = '../logs/'
DATA_DIR = '../data/processed/checkpoint/'
TB_DIR = LOG_DIR + f'tb/{args.model_name}.{datetime.now().strftime("%Y%m%d-%H%M%S")}'
MODEL_DIR = '../models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json'

NUM_EPOCHS = 10
BATCH_SIZE = 512

# Logging Setup
log.basicConfig(level=log.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                filename=os.path.join(LOG_DIR, f'training.{args.model_name}.log'))
logger = log.getLogger()

# Training Functions
class EndEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log.info(f"Epoch {epoch + 1}: Loss: {logs.get('loss')}, Custom AUC: {logs.get('custom_auc')}")

def set_tb_env():
    return TensorBoard(log_dir=TB_DIR, histogram_freq=1)

def train_model(model, dataset_id, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, tb_callback=None):
    custom_logger = EndEpochLogger()

    data_loader = DataLoader(dataset_id=dataset_id, batch_size=batch_size)
    history = model.fit(data_loader, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[custom_logger, tb_callback])
    log.info(f'Training finished. Loss: {history.history["loss"][-1]}, Custom AUC: {history.history["custom_auc"][-1]}')

# Main Function
def main(model_name):
    setup_gpus(logger)
    model = load_model(MODEL_DIR, model_name, logger)
    
    # Log the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    log.info("\n".join(model_summary))
    tb_callback = set_tb_env()
    train_model(model, 'train', args.epochs, args.batch_size, tb_callback)
    log.info('Model trained')
    
    model.save(os.path.join(MODEL_DIR, model_name))
    log.info('Model Saved.\nDone.')

# Execute Main Function
if __name__ == '__main__':
    main(args.model_name)
