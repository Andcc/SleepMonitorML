# Import functions
import logging as log
import os, sys, argparse
import tensorflow as tf
# Add custom modules
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
src_dir = os.path.normpath(src_dir)
sys.path.append(src_dir)
from misc import setup_gpus, custom_auc, load_model
from data_loader import DataLoader


# Argument Parsing
parser = argparse.ArgumentParser(description='Model Evaluation Script')
parser.add_argument('--f_name', type=str, required=True, help='Filename for the log')
parser.add_argument('--m_name', type=str, required=True, help='Model name to evaluate')
parser.add_argument('--dataset', default='dev', type=str, required=False, help='Dataset to evaluate on')
parser.add_argument('--batch_size', default='dev', type=str, required=False, help='Number of samples per batch')
args = parser.parse_args()

# Constants
LOG_NAME = os.path.abspath('../logs/' + args.f_name)
MODEL_TO_EVAL = args.m_name
DATASET = args.dataset
DATA_DIR = '../data/processed/checkpoint/'
MODEL_DIR = '../models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json'


# Logging
logger = log.getLogger()
logger.setLevel(log.DEBUG)
file_handler = log.FileHandler(LOG_NAME)
file_handler.setLevel(log.DEBUG) 
formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# Test function
def test_model(model, test_data, test_labels):
    logger.info('Testing model...')
    test_loss, test_auc = model.evaluate(test_data, test_labels, verbose=1)
    logger.info(f'Test loss: {test_loss}, Test AUC: {test_auc}')
    return test_loss, test_auc


# Main
def main():
    try: 
        setup_gpus(logger)
    except Exception as e:
        logger.error(f'Error setting up GPUs: {e}')
        raise e
    try:
        model = load_model(MODEL_DIR, MODEL_TO_EVAL, logger)
    except Exception as e:
        log.error(f'Error loading model {MODEL_TO_EVAL}: {e}')
        raise e
    
    model.summary(print_fn=lambda x: logger.info(x))
    
    logger.info(f'Loaded model {MODEL_TO_EVAL}')
    data_loader = DataLoader(dataset_id=DATASET, batch_size=512)
    test_data, test_labels = data_loader.get_eval_dataset('dev')
    logger.info(f'Loaded {DATASET} data')
    test_model(model, test_data, test_labels)
    logger.info('Done.')
    

if __name__ == '__main__':
    
    main()