import tensorflow as tf
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Data_block:
    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        self.filters = filters

    def get_tokenizer(self, text, num_words, filename=None, save=False):
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=num_words,
            filters=self.filters,
            lower=True,
            split=" ",
            oov_token=0,
        )
        tokenizer.fit_on_texts(text)
        if save:
            file = open(filename, "w")
            file = file.write(json.dumps(tokenizer.to_json()))
        return tokenizer

    def restore_tokenizer(self, filename):
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(
            json.load(open(filename, "r"))
        )
        return tokenizer


def tokenize_sentence(inp_text, **kwargs):
    data_ = Data_block()
    files = kwargs

    if "json_file" in files.keys():
        tokenizer_ = data_.restore_tokenizer(files["json_file"])
    elif ("save" and "filename") in files.keys():
        tokenizer_ = data_.get_tokenizer(
            text=inp_text,
            num_words=files["num_words"],
            save=files["save"],
            filename=files["filename"],
        )
    else:
        tokenizer_ = data_.get_tokenizer(text=inp_text, num_words=files["num_words"])
    out_seq = tokenizer_.texts_to_sequences(inp_text)
    return tokenizer_, out_seq


def seq_preprocess(seq, target_):
    assert target_ in ("source", "target")
    pad_dict = {"<bos>": 0, "<eos>": max(sum(seq, [])) + 2}

    #### custom blocks and doesn't generalize based on tasks !!!
    if target_ == "source":
        prep = [seq[t] + [pad_dict["<eos>"]] for t in range(len(seq))]
    if target_ == "target":
        prep = [
            [pad_dict["<bos>"]] + seq[t] + [pad_dict["<eos>"]] for t in range(len(seq))
        ]

    max_length = max([len(j) for j in prep])
    prep = pad_sequences(
        prep, maxlen=max_length, padding="post", value=max(sum(seq, [])) + 3
    )
    pad_dict["mask_token"] = max(sum(seq, [])) + 3
    pad_dict["length"] = max_length
    return prep, pad_dict
