Note the code scripts were given by the original authors by emails. As the given scripts are partial, we add the other essential part by ourselves. 

# Structure

# Workflow
An example of workflow is as follows:

## (1) Preprocess data
```bash
# go to src/
# save the preprocessed bugs
python preprocess.py --project eclipse
# ==> generated data ./data/preprocess/eclipse/
```

## (2) Train word2vec
```
python word2vec.py --project eclipse
```

## (3) Represent bug report as matrix
```
python toMatrix.py --project eclipse
```

## (4) Represent bug pair pair
```
python toDual.py --project eclipse
```

## (5) Train the model
```
CUDA_VISIBLE_DEVICES=3 python DCCNN.py --project eclipse
```

## (6) Evaluate the model
```
CUDA_VISIBLE_DEVICES=3 python evaluate.py --project eclipse --model_num 1
```