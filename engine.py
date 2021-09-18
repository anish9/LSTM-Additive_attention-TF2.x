import os
import tensorflow as tf
import numpy as np
import pandas as pd
from utils import data_utils,model_utils
import json


load_config = json.load(open("config.json","r"))

dataset = pd.read_csv(load_config["csv_path"])
dataset = dataset.iloc[:load_config["select"],:]
inp_tok,inputs = data_utils.tokenize_sentence(dataset[load_config["input_node"]],json_file=load_config["input_vectorizer"])
inputs,input_config = data_utils.seq_preprocess(inputs,"source")
out_tok,outputs = data_utils.tokenize_sentence(dataset[load_config["output_node"]],json_file=load_config["output_vectorizer"])
outputs,output_config = data_utils.seq_preprocess(outputs,"target")

for k,v in {y:x for x,y in output_config.items()}.items():
    out_tok.index_word[k]=v

data = tf.data.Dataset.from_tensor_slices((inputs,outputs))
data = data.shuffle(100).batch(batch_size=load_config["batch_size"],drop_remainder=True)

checkpoint_dir = load_config["ckpt_dir"]+"/"
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
optimizer = tf.keras.optimizers.Nadam(1e-3)


def run_trainer(dataset,encoder,decoder,batch,epochs,ckpt_writer,mask_token=output_config["mask_token"]):
    print("trainer started ....")
    
    for epoch in range(epochs):
        total_loss = 0
        for i,(x,y) in enumerate(dataset):
            loss = 0
            y_f = y[:,1:]
            enc_state= encoder.get_state()
            with tf.GradientTape() as t:
                enc_out,enc_state = encoder(x,enc_state)
                dec_inp = tf.reshape([[0]*batch],(batch,-1))
                for steps in range(y_f.shape[-1]):
                    dec_out,dec_state = decoder(dec_inp,enc_state,enc_out)
                    loss_value = model_utils.loss_function(y_f[:,steps],dec_out,mask_val=mask_token)
                    loss+=loss_value
                    dec_inp = tf.reshape(y_f[:,steps],(batch,-1))
            avearge_loss = loss/y_f.shape[-1]
            total_loss+=avearge_loss
            trainable_vars = encoder.variables+decoder.variables
            grads = t.gradient(loss,trainable_vars)
            optimizer.apply_gradients(zip(grads,trainable_vars))
        if epoch % load_config["save_frequency"]==0:
            print(f"EPOCH : {epoch} - total_loss : {total_loss}")
            model_utils.predict_samples(batch,encoder,decoder,x,y,inp_tok,out_tok)
        if epoch %load_config["save_frequency"]==0:
            ckpt_writer.save(checkpoint_prefix)


