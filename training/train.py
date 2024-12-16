import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import tensorflow as tf
import re
from model.architecture import LSTM_SA, customized_loss
from model.configurations import *
import argparse
import math
import keras
import pandas as pd
import random
from functools import partial

TRAIN_SEED = 1
tf.random.set_seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED) # for reproducibility

def parse_tfrecord_fn(example, channel):
    """Parses a single tf.train.Example back into the required input format."""
    feature_description = {
        'ext': tf.io.FixedLenFeature([WINDOW_SIZE * N_CHANNELS], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.int64),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    
    X_ext = tf.reshape(parsed_example['ext'], (WINDOW_SIZE, N_CHANNELS))
    y = parsed_example['label']
    X_ext = X_ext[:, channel]
    X_ext = X_ext[:, tf.newaxis]
    return X_ext, y

def load_tfrecord_dataset(tfrecord_file, channel, batch_size, shuffle=True, cache_in_memory=False, cache_file=None):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file, compression_type="GZIP")
    
    parse_fn_with_channel = partial(parse_tfrecord_fn, channel=channel)

    # Parse the dataset
    parsed_dataset = raw_dataset.map(parse_fn_with_channel, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache_in_memory: # Cache in memory
        parsed_dataset = parsed_dataset.cache()  
    elif cache_file: # Cache to a file on disk
        parsed_dataset = parsed_dataset.cache(cache_file)  
    
    #dataset_size = sum(1 for _ in parsed_dataset) if parsed_dataset.cardinality() <= 0 else parsed_dataset.cardinality().numpy()

    dataset_size = -2
    
    # Separate positive and negative samples
    positive_dataset = parsed_dataset.filter(lambda X, y: tf.equal(y[0], 1))
    negative_dataset = parsed_dataset.filter(lambda X, y: tf.equal(y[0], 0))
    
    # Create a balanced dataset with equal positives and negatives
    balanced_dataset = tf.data.Dataset.sample_from_datasets(
        [positive_dataset, negative_dataset],
        weights=[0.5, 0.5]  # Equal weight for positives and negatives
    )
    
    if shuffle:
        balanced_dataset = balanced_dataset.shuffle(buffer_size=10000, seed=TRAIN_SEED, reshuffle_each_iteration=True)
    
    # Batch and prefetch for performance
    dataset = (balanced_dataset
               .batch(batch_size, drop_remainder=True)  # Ensure full batches
               .prefetch(tf.data.AUTOTUNE))
    
    return dataset, dataset_size


def train_model(channel, train_path, val_path, checkpoint_path, first=True):
    model = LSTM_SA(lstm_units=MODEL_PARAMS[channel]['n1'], 
                    dense_units=MODEL_PARAMS[channel]['n2'], 
                    dropout_rate1=MODEL_PARAMS[channel]['p1'],
                    dropout_rate2=MODEL_PARAMS[channel]['p2'])

    if first:
        print(model.get_config())

    # Compile model
    model.compile(
        loss=LOSS_FUNCTION(from_logits=False),# customized_loss,
        optimizer=OPTIMIZER(),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.5), keras.metrics.F1Score(threshold=0.5)],
        run_eagerly=True
    )

    if first:
        print(model.summary())

    channel_num = MODEL_PARAMS[channel]['num']
    
    # Load data
    train_dataset, train_size = load_tfrecord_dataset(train_path, channel_num, batch_size=BATCH_SIZE,
                                                    shuffle=True, cache_in_memory=False, cache_file=None)
    if first:
        print(f"Training size: {train_size}")
    
    val_dataset, val_size = load_tfrecord_dataset(val_path, channel_num, batch_size=BATCH_SIZE, 
                                                shuffle=False, cache_in_memory=False, cache_file=None)
    if first:
        print(f"Validation size: {val_size}")
    
    selected_callbacks = [] # keras.callbacks.EarlyStopping(monitor='val_loss')
    # Setup checkpoint callback
    callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    
    # last_cp_path = checkpoint_path.replace(".keras", "_last.keras")
    # last_callback = keras.callbacks.ModelCheckpoint(
    #     filepath=last_cp_path,
    #     verbose=0,
    # )
    
    selected_callbacks += [callback] #, last_callback]
    
    # Train model
    model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=val_dataset,
        callbacks=selected_callbacks,
        verbose=1
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LeNet-like model for Sleep apnea detection using Sp02 and PR')

    # Add arguments
    parser.add_argument('-train', '--train_file',  type=str, help='Training data path (should endswith train.tfrecord)')
    parser.add_argument('-val', '--val_file',  type=str, help='Validation data path (should endswith val.tfrecord)')
    parser.add_argument('-cp', '--checkpoint', type=str, help='Checkpoint path (*.keras), only use "nni.keras" when run with nni')
    parser.add_argument('-ch', '--channel', type=str, help=f"Channel selection (available option: {MODEL_PARAMS.keys()})", choices=MODEL_PARAMS.keys())
    parser.add_argument('--multiple', action='store_true', help='Specify if want to run multiple training on multiple pair of (train*.tfrecord and val*.tfrecord)')

    # Parse the arguments
    args = parser.parse_args()

    if os.path.exists(args.checkpoint):
        ans = input(f"\n-----------The checkpoint already existed, do you want to override it? [Y / other:exit program] ")
        if ans.lower() != 'y':
            sys.exit(1)
    
    train_files = [args.train_file]
    val_files = [args.val_file]
    if args.multiple:
        train_files = glob.glob(args.train_file.replace(".tfrecord", "*.tfrecord"))
        train_files = sorted(train_files)
        val_files = glob.glob(args.val_file.replace(".tfrecord", "*.tfrecord"))
        val_files = sorted(val_files)
        
        assert len(train_files) == len(val_files)
    
    for i in range(len(train_files)):
        train_file = train_files[i]
        val_file = val_files[i]
        part = train_file[train_file.rfind("/") + len("train_") + 1: train_file.rfind(".tfrecord")]
        print(f"========================================================== Part {part} ({i + 1}/{len(train_files)}) ==========================================================")
        
        checkpoint_path = args.checkpoint.replace(".keras", f"_{part}_ch_{args.channel}.keras")
        train_model(args.channel, train_file, val_file, checkpoint_path, first=(i==0))

    print("-----------Completed-----------")