from tensorflow.keras.metrics import AUC
import tensorflow as tf
import os

# Custom AUC Metric
custom_auc_metric = AUC(multi_label=True, num_labels=3, label_weights=[.001, .495, .495], name='custom_auc')

def custom_auc(y_true, y_pred):
    """
    Custom AUC function.
    """
    y_true_reshaped = tf.reshape(y_true, [-1, 3])
    y_pred_reshaped = tf.reshape(y_pred, [-1, 3])
    return custom_auc_metric(y_true_reshaped, y_pred_reshaped)


# Set GPUs
def setup_gpus(logger=None):
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            raise ValueError('No GPUs found!')
        logger.info(f'GPUs found: {gpus}')
    except Exception as e:
        logger.error(f'Error setting up GPUs: {e}')
        raise e
    
# Model Loading
def load_model(model_dir, model_name, logger):
    try:
        model = tf.keras.models.load_model(os.path.join(model_dir, model_name),
                                           custom_objects={'custom_auc': custom_auc})
        return model
    except Exception as e:
        logger.error(f'Error loading model {model_name}: {e}')
        raise e