import os
import logging
import json
import cProfile
import pstats
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set seed and constants
seed = 42
np.random.seed(seed)
WINDOW_SIZE = 150
BATCH_SIZE = 50

# Directories and File paths
LOG_DIR = '../logs/'
CHECKPOINT_DIR = '../data/processed/checkpoint/'
DATA_DIR = '../data/raw/'
PROCESSED_DATA_DIR = '../data/processed/transformer_data'
LOG_NAME = 'data_processing.v3.2.log'

# Logging configuration
logging.basicConfig(filename=os.path.join(LOG_DIR, LOG_NAME),
                    level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Helper functions


def load_data(file_name, set_index='series_id'):
    logging.info(f'Loading {file_name}')
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path).set_index(set_index) if file_name.endswith('.csv') else pd.read_parquet(file_path).set_index(set_index)


def save_data(series_id, sequences, labels, events):
    if sequences:  # Check if sequences is not empty
        logging.info(f'Saving data for series {series_id}')
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{series_id}.npz')
        np.savez_compressed(checkpoint_path, sequences=np.array(
            sequences), labels=np.array(labels), events=events)
    else:
        logging.warning(f'No sequences to save for series {series_id}')


def load_all_processed_data():
    logging.info('Loading all processed data')
    sequences, labels, events = [], [], []
    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(
                CHECKPOINT_DIR, filename), allow_pickle=True)
            sequences.append(data['sequences'])
            labels.append(data['labels'])
            events.extend(data['events'])
    return np.concatenate(sequences), np.concatenate(labels), events


def sample_and_inspect(sequences, labels, events, num_samples=5):
    import random

    print(f'Sequences shape = {sequences.shape}')
    print(f'Labels shape = {labels.shape}')
    print(f'Number of event lists = {len(events)}\n')

    sample_indices = random.sample(range(len(sequences)), num_samples)
    for i in sample_indices:
        label_sample = labels[i] if i < len(labels) else "No label"
        event_sample = events[i] if i < len(events) else "No event"

        print(f"Sample Index: {i}")
        print(f"Sequence shape: {sequences[i].shape}")
        print(f"Label: {len(label_sample)}")
        print(f"Events: {len(event_sample)}\n")


def check_series_processed(series_id, submit_series=['038441c925bb', '03d92c9f6f8a', '0402a003dae9']):
    if series_id in submit_series:
        path_submit = os.path.join(CHECKPOINT_DIR, f'{series_id}-submit.npz')
        path_train = os.path.join(CHECKPOINT_DIR, f'{series_id}.npz')
        return os.path.exists(path_train) and os.path.exists(path_submit)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{series_id}.npz')
    return os.path.exists(checkpoint_path)


def log_series_start(series_id, series_data):
    logging.info(
        f'Starting processing series {series_id}, size: {len(series_data)} at {datetime.now()}')


def log_series_end(series_id, valid_windows):
    logging.info(
        f'Finished processing series {series_id} with {valid_windows} valid windows at {datetime.now()}')


def profile_and_save(pr, filename=f"../data/processed/profiling/{'.'.join(LOG_NAME.split('.')[0:2])}.profile_results.prof"):
    stats = pstats.Stats(pr)
    stats.dump_stats(filename)


# Core Processing Functions


def generate_sequences(df, train_events=None):
    logging.info('Generating sequences')
    is_train = train_events is not None
    sequences, labels, events = [], [], []
    for series_id, series_data in df.groupby(level=0):
        log_series_start(series_id, series_data)
        if check_series_processed(series_id):
            logging.info(f'Series {series_id} already processed. Skipping.')
            continue

        try:
            valid_windows = 0
            if is_train and series_id in train_events.index:
                for _, event_row in train_events.loc[series_id].iterrows():
                    event_step = event_row['step']
                    if pd.isna(event_step):
                        logging.warning(
                            f'NaN event step in series {series_id}')
                        continue
                    for pos in range(WINDOW_SIZE):
                        start_step = max(event_step - pos, 0)
                        end_step = min(start_step + WINDOW_SIZE,
                                       len(series_data))
                        try:
                            window = series_data[(series_data['step'] >= start_step) &
                                                 (series_data['step'] < end_step)]
                            if valid_window(window, train_events):
                                valid_windows += 1
                                sequences.append(
                                    window[['anglez', 'enmo']].values)
                                labels.append(generate_labels(
                                    series_id, window, train_events))
                                events.append(
                                    generate_events(series_id, window))
                        except IndexError as idx_err:
                            logging.error(
                                f'Window creation error in series {series_id} at start step {start_step} and end step {end_step}: {idx_err}')

            else:
                logging.info('No train events. Generating single window')
                series_id += '-submit'
                print(series_data)
                if len(series_data) == WINDOW_SIZE:
                    sequences.append(series_data[['anglez', 'enmo']].values)
                    labels.append(np.zeros(WINDOW_SIZE))
                    events.append(generate_events(series_id, series_data))
                    valid_windows = 1

            save_data(series_id, sequences, labels, events)
            sample_and_inspect(np.array(sequences),
                               np.array(labels), events, 1)
            sequences, labels, events = [], [], []
        except MemoryError as mem_err:
            logging.error(
                f'Memory error processing series {series_id}: {mem_err}')
        except Exception as e:
            logging.error(f'Error processing series {series_id}: {e}')
        finally:
            log_series_end(series_id, valid_windows)
    logging.info('Finished generating sequences')


def valid_window(window, train_events):
    if window.shape[0] != WINDOW_SIZE or not all(window['step'].values[i] + 1 == window['step'].values[i + 1] for i in range(len(window) - 1)):
        return False
    if train_events is not None:
        series_id = window.index[0]
        window_steps = set(window['step'].values)
        event_steps = set(
            train_events.loc[series_id]['step'].values) if series_id in train_events.index else set()
        return not window_steps.isdisjoint(event_steps)

    return True


def generate_labels(series_id, window, train_events):
    label = np.zeros(WINDOW_SIZE)
    if series_id in train_events.index:
        event_steps = train_events.loc[series_id, 'step']
        event_steps = event_steps if isinstance(
            event_steps, pd.Series) else pd.Series([event_steps])

        for i, (_, row) in enumerate(window.iterrows()):
            if row['step'] in event_steps.values:
                event = train_events.loc[(train_events.index == series_id) & (
                    train_events['step'] == row['step']), 'event'].iloc[0]
                label[i] = 1 if event == 'onset' else 2 if event == 'wakeup' else 0
    return label


def generate_events(series_id, window):
    return [f'{series_id}_{row["step"]}_{row["timestamp"]}' for _, row in window.iterrows()]


def process_all_sequences(train_size=0.8):
    logging.info('Processing all sequences')
    distribution = {'train': 0, 'dev': 0, 'test': 0}
    observed_sequences = {'train': set(), 'dev': set(),
                          'test': set(), 'submission': set()}
    logging.info(f'Initial distribution: {distribution}')

    for filename in os.listdir(CHECKPOINT_DIR):
        if filename.endswith('.npz'):
            sequence_id = filename.split('.')[0]
            if sequence_id not in set().union(*observed_sequences.values()):
                distribution, observed_sequences = allocate_and_save_sequence(
                    filename, distribution, observed_sequences, train_size)
            else:
                logging.info(
                    f'Series {sequence_id} already processed. Skipping.')
    logging.info('Finished allocating all sequences to a dataset')
    return distribution, observed_sequences


def allocate_and_save_sequence(filename, distribution, observed_sequences, train_size=0.7):
    sequence_id = filename.split('.')[0]
    if 'submit' in sequence_id:
        dataset = 'submission'
    else:
        total_sequences = sum(distribution.values())
        target_ratios = {'train': train_size, 'dev': (
            1 - train_size) / 2, 'test': (1 - train_size) / 2}
        dataset = min(target_ratios.keys(), key=lambda k: (
            distribution[k] / total_sequences if total_sequences else 0) - target_ratios[k])

        distribution[dataset] += len(np.load(os.path.join(
            CHECKPOINT_DIR, filename), allow_pickle=True)['sequences'])
    observed_sequences[dataset].add(sequence_id)
    logging.info(
        f'Allocated {sequence_id} to {dataset}.\nUpdated distribution: {distribution}')

    return distribution, observed_sequences


def save_allocation_records(observed_sequences, disbribution):
    logging.info(
        f'Saving allocation records.\nThe data is split as follows:\n{disbribution}')
    allocation_path = os.path.join(
        PROCESSED_DATA_DIR, 'allocation_records.json')
    with open(allocation_path, 'w') as file:
        json.dump({k: list(v) for k, v in observed_sequences.items()}, file)


def split_and_save_data(sequences, labels, events, save_dir='../data/processed/transformer_data', train_size=0.7, seed=seed, submit=False):
    logging.info('Splitting and saving data')
    os.makedirs(save_dir, exist_ok=True)

    if submit:
        np.save(os.path.join(save_dir, 'submission/x_submit.npy'), sequences)
        np.save(os.path.join(save_dir, 'submission/events_submit.npy'), events)
        logging.info('Finished splitting and saving submit data')
        return

    X_train, X_temp, y_train, y_temp, events_train, events_temp = train_test_split(
        sequences, labels, events, train_size=train_size, random_state=seed
    )
    X_dev, X_test, y_dev, y_test, events_dev, events_test = train_test_split(
        X_temp, y_temp, events_temp, test_size=0.5, random_state=seed
    )

    np.save(os.path.join(save_dir, 'train/X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'train/y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'train/events_train.npy'), events_train)
    logging.info('Finished splitting and saving train data')
    np.save(os.path.join(save_dir, 'dev/X_dev.npy'), X_dev)
    np.save(os.path.join(save_dir, 'dev/y_dev.npy'), y_dev)
    np.save(os.path.join(save_dir, 'dev/events_dev.npy'), events_dev)
    logging.info('Finished splitting and saving dev data')
    np.save(os.path.join(save_dir, 'test/X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'test/y_test.npy'), y_test)
    np.save(os.path.join(save_dir, 'test/events_test.npy'), events_test)
    logging.info('Finished splitting and saving test data')


# Main Function
def main():
    train_series = load_data('train_series.parquet')
    train_events = load_data('train_events.csv')
    test_series = load_data('test_series.parquet')

    generate_sequences(train_series, train_events)
    generate_sequences(test_series, None)

    distribution, observed_sequences = process_all_sequences(train_size=0.8)

    save_allocation_records(observed_sequences, distribution)

    logging.info('Script run successfully')


if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    try:
        main()
    except Exception as e:
        logging.error(f'Error in main: {e}')
    finally:
        pr.disable()
        profile_and_save(pr)
