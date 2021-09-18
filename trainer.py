import os
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import data_utils, model_utils
from tensorflow.keras import layers
import json
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from engine import load_config, inp_tok, out_tok, run_trainer
from engine import checkpoint_dir, data

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


vocab_enc = max(inp_tok.index_word.keys()) + 5
vocab_dec = max(out_tok.index_word) + 5
units = load_config["units"]
embed_dim = load_config["embed_size"]
BATCH_SIZE = load_config["batch_size"]
EPOCHS = load_config["epochs"]
ENCODER_MODEL = model_utils.Encoder(
    lstm_units=units, embed_dim=embed_dim, vocab=vocab_enc, batch_size=BATCH_SIZE
)
DECODER_MODEL = model_utils.Decoder(
    lstm_units=units, embed_dim=embed_dim, vocab=vocab_dec, batch_size=BATCH_SIZE
)
OPTIMIZER = tf.keras.optimizers.Nadam(1e-3)
DATASET = data

checkpoint = tf.train.Checkpoint(
    optimizer=OPTIMIZER, encoder=ENCODER_MODEL, decoder=DECODER_MODEL
)
summary_writer = tf.summary.create_file_writer(logdir=checkpoint_dir)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

if __name__ == "__main__":
    run_trainer(DATASET, ENCODER_MODEL, DECODER_MODEL, BATCH_SIZE, EPOCHS, checkpoint)
