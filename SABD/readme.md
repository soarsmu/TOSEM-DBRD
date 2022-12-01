# SABD (including Siamese Pair, SABD)
Note the original repo is [here](https://github.com/irving-muller/soft_alignment_model_bug_deduplication). All credit to the original authors.

## SABD

We show an example of workflow to run SABD on Mozilla

### (1) Create datasets
```bash
python data/create_dataset_our_methodology_json.py --database mozilla  --bug_data ../dataset/mozilla/mozilla.json --training ../dataset/mozilla/training_split_mozilla.txt --validation ../dataset/mozilla/validation_mozilla.txt --test ../dataset/mozilla/test_mozilla.txt --date="2020/01/01" --no_tree --dev_perc=0.05
```

### (2) clean data
```bash
python data/clean_data.py --bug_dataset ../dataset/mozilla/mozilla.json --output ../dataset/mozilla/mozilla_soft_clean.json --fields short_desc description --type soft --rm_punc --lower_case --rm_number --stop_words --stem --rm_char
```

### (3) generate pairs
### generating training
```bash
python data/generate_pairs_triplets.py --bug_data ../dataset/mozilla/mozilla.json --dataset ../dataset/mozilla/training_split_mozilla.txt --n 1 --type random
```

### generating validation
```bash
python data/generate_pairs_triplets.py --bug_data ../dataset/mozilla/mozilla.json --dataset dataset/mozilla/validation_mozilla.txt --n 1 --type random
```

### (4) union train_split and valid to train
```bash
python util/join_training_validation.py -ts ../dataset/mozilla/training_split_mozilla.txt -tsp ../dataset/mozilla/training_split_mozilla_pairs_random_1.txt -v ../dataset/mozilla/validation_mozilla.txt -vp ../dataset/mozilla/validation_mozilla_pairs_random_1.txt -t ../dataset/mozilla/training_mozilla.txt  -tp ../dataset/mozilla/training_mozilla_pairs_random_1.txt
```

### (5) generate categorical lexicon
```bash
python data/generate_categorical_lexicon.py --bug_data ../dataset/mozilla/mozilla.json -o ../dataset/mozilla/categorical_lexicons.json
```

### (6) generate glove embeddings
```bash
python util/create_dataset_word_embedding_json.py --training ../dataset/mozilla/training_split_mozilla.txt --bug_database ../dataset/mozilla/mozilla.json --output word_embedding/mozilla/mozilla_soft_clean.txt --clean_type soft --rm_punc --lower_case --rm_number --stop_words --stem --rm_char
```

### (7) make glove embeddings
go to glove folder, run ./demo.sh <= modifiy the ./demo.sh to the corret ITS

### (8) merge two word2vec files
```bash
python util/merge_wv_files.py --main word_embedding/glove.42B.300d.txt --aux word_embedding/mozilla/glove_300d_mozilla.txt --output word_embedding/mozilla/glove_42B_300d_mozilla_soft.txt
```

### (9) transform txt to binary
```bash
python data/transform_glove_binary.py ../word_embedding/mozilla/glove_42B_300d_mozilla_soft.txt ../word_embedding/mozilla/glove_42B_300d_mozilla_soft.npy ../word_embedding/mozilla/glove_42B_300d_mozilla_soft.lxc -db ../dataset/mozilla/mozilla_soft_clean.json -tk white_space -filters TransformLowerCaseFilter
```

### (10) run SABD
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/sabd.py -F ../experiments with ../json_parameters/sabd_mozilla.json "recall_rate.window=365"
```

----

- Siamese Pair


```base
CUDA_VISIBLE_DEVICES=0 python experiments/siamese_pairs.py -F ../experiments with ../json_parameters/pair_mozilla.json "recall_rate.window=365"
```

- glove

This subfolder is downloaded from [stanfordnlp/glove](https://github.com/stanfordnlp/GloVe/).