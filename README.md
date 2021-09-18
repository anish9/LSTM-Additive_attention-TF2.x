# seq-to-seq-tensorflow-2.x
### There are certainly simplified two steps to start training:
* Build config.
* Start trainer.

##### For demo purpose, we will create english-tamil language translator. 
<p align="center">
  <img width="800" height="110" src="https://github.com/anish9/seq-to-seq-tensorflow-2.x/blob/main/assets/data.png">
</p>

* Build the tokenizer for the dataset, the structure of the dataset is previewed in the above image.
```
python build_config.py --csv_path datasets/english_tamil.csv --input_id question --output_id answer 
--data_count 300 --ckpt_dir eng_tam --save_frequency 25 --epochs 400

Arguments short explanation:
-input id   : column name of source values
-output_id  : column name of target values
-data_count : Number of rows to be considered for training

```
* A config json file will be auto-generated with training config. we can edit the fields if needed.
```
{
    "csv_path": "datasets/english_tamil.csv",
    "select": 300,
    "input_node": "question",
    "output_node": "answer",
    "input_vectorizer": "question.json",
    "output_vectorizer": "answer.json",
    "batch_size": 128,
    "epochs": 400,
    "units": 64,
    "embed_size": 32,
    "ckpt_dir": "eng_tam",
    "save_frequency": 25
}
```
* Onece the config is generated, execute trainer.py
```
python trainer.py
```
* Testing
```
inference.ipynb notebook can be used for testing the model
```
<p align="center">
  <img width=840" height="180" src="https://github.com/anish9/seq-to-seq-tensorflow-2.x/blob/main/assets/pred.png">
</p>







