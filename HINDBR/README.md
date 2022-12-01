Note that the original repo is [here](https://github.com/hindbr/HINDbr). 
All credit to the original authors. 
We use the subset of this repo and modifited based on their code.

# Workflow
An example of workflow is as follows:
## (1) Bug Report HIN Construction
```bash
python3 data_generation_br_hin.py --project eclipse
```

## (2) Generating Bug Groups
```bash
python3 data_generation_bug_groups.py --project eclipse-old
```

## (3)
### Train word2vec
```
python word2vec_training.py --project eclipse
```

### Train hin2vec
```
python hin2vec_training.py --project eclipse
```

## (4) Generating training pairs
```
python data_generation_model_training.py --project vscode
```

## (5) Generating training data
```
python data_generation_training.py --project eclipse
```

## (6) Train model
```
CUDA_VISIBLE_DEVICES=1 python models.py --project eclipse --variant text
```

## (7) Evaluate model
```
CUDA_VISIBLE_DEVICES=0 python evaluate.py --project eclipse --variant text --model_num 1
```