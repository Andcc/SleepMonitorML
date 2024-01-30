# Import functions
import argparse
import logging as log
import os
import tensorflow as tf
from modeling import setup_gpus, load_all_data, one_hot_encode_labels, TransformerBlock, custom_auc

# Argument Parsing
parser = argparse.ArgumentParser(description='Model Evaluation Script')
parser.add_argument('--f_name', type=str, required=True, help='Filename for the log')
parser.add_argument('--m_name', type=str, required=True, help='Model name to evaluate')
parser.add_argument('--dataset', default='dev', type=str, required=False, help='Dataset to evaluate on')
args = parser.parse_args()


# Constants
LOG_NAME = os.path.abspath('../logs/' + args.f_name)
MODEL_TO_EVAL = args.m_name
DATASET = args.dataset

DATA_DIR = '../data/processed/checkpoint/'
MODEL_DIR = '../data/models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json'


# Logging
logger = log.getLogger()
logger.setLevel(log.DEBUG)
file_handler = log.FileHandler(LOG_NAME)
file_handler.setLevel(log.DEBUG) 
formatter = log.Formatter('%(asctime)s %(levelname)s %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)




def test_model(model, test_data, test_labels):
    logger.info('Testing model...')
    test_loss, test_auc = model.evaluate(test_data, test_labels, verbose=1)
    logger.info(f'Test loss: {test_loss}, Test AUC: {test_auc}')
    return test_loss, test_auc



# Main
def main():
    try: 
        setup_gpus()
    except Exception as e:
        logger.error(f'Error setting up GPUs: {e}')
        raise e
    try:
        model = tf.keras.models.load_model(MODEL_DIR + MODEL_TO_EVAL, custom_objects={'custom_auc': custom_auc, 
                                                                                      'TransformerBlock': TransformerBlock
                                                                                      })
    except Exception as e:
        log.error(f'Error loading model {MODEL_TO_EVAL}: {e}')
        raise e
    
    model.summary(print_fn=lambda x: logger.info(x))
    
    logger.info(f'Loaded model {MODEL_TO_EVAL}')
    test_data, test_labels, _ = load_all_data(DATASET)
    logger.info(f'Loaded {DATASET} data')
    test_labels = one_hot_encode_labels(test_labels)
    test_model(model, test_data, test_labels)
    logger.info('Done.')
    

if __name__ == '__main__':
    
    main()