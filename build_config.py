import argparse
import pandas as pd
from utils import data_utils
import json

#python build_tokenizer.py --csv_path datasets/friends.csv --input_id question --output_id answers --data_count 5000
train_config = dict()

parser = argparse.ArgumentParser()

parser.add_argument("--csv_path",  type=str, required=True, help="csv file path")
parser.add_argument("--input_id",  type=str, required=True, help="input column name from CSV")
parser.add_argument("--output_id", type=str, required=True, help="output column name from CSV")
parser.add_argument("--data_count", type=int, required=True, help="total rows to be selected")
parser.add_argument("--batchsize", type=int, default=128,required=False, help="batchsize for training")
parser.add_argument("--epochs", type=int, default=60,required=False, help="epochs to train")
parser.add_argument("--units", type=int, default=64,required=False, help="Recurrent number of cells ex:(lstm units)")
parser.add_argument("--embeds", type=int, default=32,required=False, help="embedding vector length")
parser.add_argument("--ckpt_dir", type=str,required=True, help="checkpoint storing directory")
parser.add_argument("--save_frequency", type=int,default=10,required=False, help="checkpoint save frequency")

args = vars(parser.parse_args())

filename = args["csv_path"]
dataset = pd.read_csv(filename)
if isinstance(args["data_count"],int):
    dataset = dataset.iloc[:args["data_count"],:]
else:
    dataset = dataset.iloc[:,:]

# saving a tokenizer
inp_tok,inputs = data_utils.tokenize_sentence(dataset[args["input_id"]],save=True,filename=args["input_id"]+".json",num_words=16000)
inp_tok,inputs = data_utils.tokenize_sentence(dataset[args["output_id"]],save=True,filename=args["output_id"]+".json",num_words=16000)

train_config["csv_path"]=args["csv_path"]
train_config["select"]=args["data_count"]
train_config["input_node"]  = args["input_id"]
train_config["output_node"] = args["output_id"]
train_config["input_vectorizer"] = args["input_id"]+".json"
train_config["output_vectorizer"] = args["output_id"]+".json"
train_config["batch_size"] = args["batchsize"]
train_config["epochs"] = args["epochs"]
train_config["units"] = args["units"]
train_config["embed_size"] = args["embeds"]
train_config["ckpt_dir"] = args["ckpt_dir"]
train_config["save_frequency"] = args["save_frequency"]

with open("config.json","w") as config_file:
    config_file.write(json.dumps(train_config,indent=4))
config_file.close()


print("Tokenizer created succesfully")