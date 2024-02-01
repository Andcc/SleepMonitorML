import os, json
import tensorflow as tf
import numpy as np

# Constants
DATA_DIR = '../data/processed/checkpoint/'
MODEL_DIR = '../data/models/'
DATASET_ALLOCATIONS = '../data/processed/transformer_data/allocation_records.json' 

# Data Loader class
class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset_id, batch_size=32, shuffle=True, data_dir=DATA_DIR, dataset_allocations=DATASET_ALLOCATIONS, noise_level=0):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.dataset_allocations = dataset_allocations

        self.series = self._get_series_ids()
        self.series_lens = {}
        self.tot_obs = None

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.on_epoch_end()

    def _get_series_ids(self):
        with open(self.dataset_allocations, 'r') as f:
            allocations = json.load(f)
            return allocations[self.dataset_id]

    def __len__(self):
        if not self.tot_obs:
            self.tot_obs = 0
            for series_id in self.series:
                with np.load(os.path.join(self.data_dir, series_id + '.npz'), allow_pickle=True) as data:
                    self.tot_obs += len(data['sequences'])
        return int(np.ceil(self.tot_obs / self.batch_size))
    
    def __getitem__(self, idx):
        obs_remain = self.batch_size
        idx = idx % len(self.series)
        x, y = [], []
        while obs_remain > 0:
            try:
                series_id = self.series[idx]
                with np.load(os.path.join(self.data_dir, series_id + '.npz'), allow_pickle=True) as data:
                    series_x, series_y, _ = data['sequences'], data['labels'], data['events']
                    series_x += np.random.normal(0, self.noise_level, series_x.shape)
                    subtract = min(obs_remain, len(series_x))
                    x.append(series_x[:subtract,:,:])
                    y.append(series_y[:subtract,:])
                    obs_remain -= subtract
                    idx += 1
            except Exception as e:
                print(e)
                print(f"Error loading series at index {idx}")
                print(f"Series shape: {len(self.series)}")
                break
                
        x = np.concatenate(x, axis=0)
        y = tf.keras.utils.to_categorical(np.concatenate(y, axis=0), num_classes=3)
        
        return x, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.series)

    def get_eval_dataset(self, dataset_id):
        with open(DATASET_ALLOCATIONS, 'r') as f:
            allocations = json.load(f)
            series_ids = allocations[dataset_id]
            all_x, all_y = [], []
            for series_id in series_ids:
                data = np.load(os.path.join(DATA_DIR, series_id + '.npz'), allow_pickle=True)
                x, y, _ = data['sequences'], data['labels'], data['events']
                all_x.append(x)
                all_y.append(y)
            return np.concatenate(all_x, axis=0), tf.keras.utils.to_categorical(np.concatenate(all_y, axis=0), num_classes=3)
