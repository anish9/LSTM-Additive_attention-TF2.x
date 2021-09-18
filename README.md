# seq-to-seq-tensorflow-2.x





### There are 2 steps to be followed to create a model(any seq-to-seq text tasks)

##### For demo, we will create english-tamil language translator

* Build a tokenizer for custom dataset using the below code
```
python build_config.py --csv_path datasets/english_tamil.csv --input_id question --output_id answer --data_count 300

Arguments short explanation:
-input id   : column name of source values
-output_id  : column name of target values
-data_count : Number of rows to be considered for training

```
* a config json file will be auto-generated with training config. we can edit the fields if we need.
```
{
    "csv_path": "datasets/english_tamil.csv",
    "select": 300,
    "input_node": "question",
    "output_node": "answer",
    "input_vectorizer": "question.json",
    "output_vectorizer": "answer.json",
    "batch_size": 128,
    "epochs": 60, 
    "units": 64, #lstm units
    "embed_size": 32 #word embedding dimension
}
```
