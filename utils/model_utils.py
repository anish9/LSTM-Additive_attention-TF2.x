import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from utils import data_utils


class Encoder(tf.keras.models.Model):
    def __init__(self, lstm_units, embed_dim, vocab, batch_size):
        super(Encoder, self).__init__()
        self.lstm_units = lstm_units
        self.embed_dim = embed_dim
        self.vocab = vocab
        self.batch_size = batch_size

        self.embed = layers.Embedding(self.vocab, self.embed_dim)
        self.drop1 = layers.Dropout(0.1)
        self.cell1 = layers.LSTM(
            self.lstm_units, return_sequences=True, return_state=True
        )
        self.drop2 = layers.Dropout(0.05)
        self.cell2 = layers.LSTM(
            self.lstm_units, return_sequences=True, return_state=True
        )

    def call(self, x, state):
        x = self.embed(x)
        x = self.drop1(x)
        x = self.cell1(x, state)
        o, h, c = self.cell2(x)
        return o, h

    def get_state(self):
        return [
            tf.zeros((self.batch_size, self.lstm_units)),
            tf.zeros((self.batch_size, self.lstm_units)),
        ]


class Decoder(tf.keras.models.Model):
    def __init__(self, lstm_units, embed_dim, vocab, batch_size):
        super(Decoder, self).__init__()
        self.lstm_units = lstm_units
        self.embed_dim = embed_dim
        self.vocab = vocab
        self.drop1 = layers.Dropout(0.1)
        self.drop2 = layers.Dropout(0.1)
        self.batch_size = batch_size
        self.fc1 = layers.Dense(self.embed_dim, activation="relu")
        self.fc2 = layers.Dense(self.embed_dim, activation="relu")
        self.fc3 = layers.Dense(self.vocab)
        self.embed = layers.Embedding(self.vocab, self.embed_dim)
        self.cell1 = layers.LSTM(
            self.lstm_units, return_sequences=True, return_state=True
        )
        self.cell2 = layers.LSTM(
            self.lstm_units, return_sequences=True, return_state=True
        )

    def attention_module(self, x, y):
        a_xy = tf.tanh(x + y)
        a_xy = tf.nn.softmax(a_xy, axis=1) * y
        a_xy = tf.expand_dims(tf.reduce_sum(a_xy, axis=1), axis=1)
        return a_xy

    def create_context(self, x, hidden_, out_):
        attend_ = self.attention_module(hidden_, out_)
        embed_ = self.embed(x)
        context_ = tf.concat((attend_, embed_), axis=-1)
        return context_

    def call(self, x, hidden_, out_):
        out_ = self.fc1(out_)
        hidden_ = self.fc2(hidden_[:, tf.newaxis, :])
        context_ = self.create_context(x, hidden_, out_)
        context_ = self.drop1(context_)
        out = self.cell1(context_)
        out, h, _ = self.cell2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = tf.reshape(out, (-1, out.shape[-1]))
        return out, h


def loss_function(real, pred, mask_val):
    mask = 1 - np.equal(real, mask_val)
    loss_ = (
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    )
    return tf.reduce_mean(loss_)


def predict_samples(batch, ENCODER_MODEL, DECODER_MODEL, inp, tar, inp_tok, out_tok):
    # train time callback inferene
    choice = np.random.randint(0, batch, 1)[0]
    question_list = []
    answer_list = []
    target_list = []
    for jj in inp[choice].numpy():
        try:
            question_list.append(inp_tok.index_word[jj])
        except:
            pass
    input_sent = " ".join(question_list)
    inp = tf.tile(inp[choice][np.newaxis, :], (batch, 1))

    tar_f = tar[:, 1:]
    for jj in tar_f[choice].numpy():
        try:
            answer_list.append(out_tok.index_word[jj])
        except:
            pass
    answer_list = [j for j in answer_list if j != "mask_token"]
    target_sent = " ".join(answer_list)
    print("input_query : ", input_sent)
    print("targe_query : ", target_sent)
    enc_val_state = ENCODER_MODEL.get_state()
    enc_val_out, enc_val_state = ENCODER_MODEL(inp, enc_val_state)
    dec_val_inp = tf.reshape([[0] * batch], (batch, -1))
    for steps in range(30):
        dec_val_out, dec_val_state = DECODER_MODEL(
            dec_val_inp, enc_val_state, enc_val_out
        )
        predword = np.argmax(dec_val_out)
        dec_val_inp = tf.reshape([[predword] * batch], (batch, -1))
        if out_tok.index_word[predword] == "<eos>":
            break
        target_list.append(out_tok.index_word[predword])
    target_list = [j for j in target_list if j != "<eos>"]
    target_sent = " ".join(target_list)
    print("pred_qury : ", target_sent)
    print("----------------------------------------------------")


def inference(
    sent, infer_enc, infer_dec, input_config, output_config, json_file, out_tok, batch=1
):
    # inference test time
    try:
        i_state_enc = infer_enc.get_state()
        question_inp = sent  # "I was out"
        _, inputs_ = data_utils.tokenize_sentence([question_inp], json_file=json_file)
        inputs_, _ = data_utils.seq_preprocess(inputs_, "source")
        inputs_[0][-1] = input_config["<eos>"]
        inputs_ = tf.keras.preprocessing.sequence.pad_sequences(
            inputs_,
            maxlen=input_config["length"],
            padding="post",
            value=input_config["mask_token"],
        )
        enc_out, enc_cstate = infer_enc(inputs_, i_state_enc)
        dec_cstate = enc_cstate
        sentence = ""
        dec_inp = tf.reshape([[0] * batch], (batch, -1))
        for u in range(120):
            dec_cout, dec_cstate = infer_dec(dec_inp, dec_cstate, enc_out)
            #             out = np.argmax(dec_cout)
            dummy = np.argsort(dec_cout)
            out = dummy[0][-1:][0]
            if out == output_config["<eos>"]:
                break
            sentence += out_tok.index_word[out] + " "
            dec_inp = tf.reshape(out, (-1, 1))
            dec_cstate = dec_cstate
        return sent, sentence
    except Exception as e:
        return sent, "sorry i didn't get you !!! (embedded word invalid)"
