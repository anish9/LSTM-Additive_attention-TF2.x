# seq-to-seq-tensorflow-2.x





### There are 2 steps to be followed to create a model(any seq-to-seq text tasks)
### For demo, we will create englih -tamil language translator

* Build a tokenizer for custom dataset using the below code
```
python build_config.py --csv_path datasets/english_tamil.csv --input_id question --output_id answer --data_count 300

Arguments short explanation:
-input id   : column name of source values
-output_id  : column name of target values
-data_count : Number of rows to be considered for training

```
