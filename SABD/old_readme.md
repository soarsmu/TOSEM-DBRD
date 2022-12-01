#################### Copied and modified from https://github.com/irving-muller/soft_alignment_model_bug_deduplication Credit to them ###############

# A Soft Alignment Model for Bug Deduplication

By Irving Muller Rodrigues, Daniel Aloise, Eraldo Rezende Fernandes, and Michel Dagenais

[Preprint](https://irving-muller.github.io/papers/MSR2020.pdf)

## Introduction

We propose a Soft Alignment Model for Bug Deduplication (SABD). 
For a given pair of possibly duplicate reports, the attention mechanism computes interdependent representations for each report, which is more powerful than previous approaches
Our experimental results demonstrate that SABD outperforms state-of-the-art systems and strong baselines in different scenarios. 

## Install

Install the following packages:

```bash
# CPU
conda install pytorch torchvision cpuonly -c pytorch 
# GPU
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

conda install -c anaconda nltk
conda install -c anaconda scipy
conda install -c anaconda ujson
conda install -c pytorch ignite=0.1.2
pip install sacred 
```

## Data

The data used in the paper can be found [here](https://zenodo.org/record/3922012). One folder contain the word embeddings and four remaining ones 
contain the dataset of the open-sources projects (Eclipse, Netbeans, Open Office and Firefox).
This data came from the [Lazar's work](https://dl.acm.org/doi/abs/10.1145/2597073.2597128).
The original dataset is available [here](http://alazar.people.ysu.edu/msr14data/).

*The commands below generates the dataset of each open-sources projects:*
> Note: Lazar's dataset has to be dumped into mongo before running these commands

```bash 
# Mozilla
python data/create_dataset_our_methodology.py --database mozilla --collection mozall --bug_data DATASET_DIR/mozilla_2001-2009_2010/mozilla_initial.json --training DATASET_DIR/mozilla_2001-2009_2010/training_split_mozilla.txt --validation  DATASET_DIR/mozilla_2001-2009_2010/validation_mozilla.txt --test DATASET_DIR/mozilla_2001-2009_2010/test_mozilla.txt --date="2010/01/01" --date_threshold="2010/12/31" --no_tree --dev_perc=0.05
    
#Eclipse
python data/create_dataset_our_methodology.py --database eclipse --collection initial --bug_data DATASET_DIR/eclipse_2001-2007_2008/eclipse_initial.json --training  DATASET_DIR/eclipse_2001-2007_2008/training_split_eclipse.txt --validation  DATASET_DIR/eclipse_2001-2007_2008/validation_eclipse.txt --test DATASET_DIR/eclipse_2001-2007_2008/test_eclipse.txt --date="2008/01/01" --date_threshold="2008/12/31" --no_tree --dev_perc=0.05

#Netbeans
python data/create_dataset_our_methodology.py --database netBeans --collection initial --bug_data DATASET_DIR/netbeans_2001-2007_2008/netbeans_initial.json --training  DATASET_DIR/netbeans_2001-2007_2008/training_split_netbeans.txt --validation  DATASET_DIR/netbeans_2001-2007_2008/validation_netbeans.txt --test DATASET_DIR/netbeans_2001-2007_2008/test_netbeans.txt --date="2008/01/01" --date_threshold="2008/12/31" --no_tree --dev_perc=0.05

#OpenOffice
python data/create_dataset_our_methodology.py --database openOffice --collection initial --bug_data DATASET_DIR/open_office_2001-2008_2010/open_office_initial.json --training  DATASET_DIR/open_office_2001-2008_2010/training_split_open_office.txt --validation  DATASET_DIR/open_office_2001-2008_2010/validation_open_office.txt --test DATASET_DIR/open_office_2001-2008_2010/test_open_office.txt --date="2008/01/01" --date_threshold="2010/12/31" --no_tree --dev_perc=0.05
```
 
*An example to how preprocess report data is shown below:*
    
```bash
python data/clean_data.py --bug_dataset DATASET_DIR/netbeans_2001-2007_2008/netbeans_initial.json --output DATASET_DIR/netbeans_2001-2007_2008/netbeans_soft_clean_rm_punc_sent_tok.txt.json --fields short_desc description --type soft --rm_punc --sent_tok --lower_case
```
    
*An example to generate pairs and triplets for training is shown below:*
    
```bash
python data/create_dataset_our_methodology.py --bug_data DATASET_DIR/open_office_2001-2008_2010/open_office_initial.json --dataset DATASET_DIR open_office_2001-2008_2010/training_open_office.txt --n 1 --type random
```
    

*An example to generate categorical lexicon is shown below:*

```bash
python data/generate_categorical_lexicon.py --bug_data DATASET_DIR/mozilla_2001-2009_2010/mozilla_initial.json -o DATASET_DIR/dataset/sun_2011/mozilla_2001-2009_2010/categorical_lexicons.json
```
        
In order to compare SABD with REP and BM25F_ext, we had to modify a little bit the original code of REP and BM25F_ext.
The modified source code can be found [here](https://github.com/irving-muller/fast-dbrd-modified).       
        
> Note: check util/create_dataset_word_embedding.py regarding the word embedding training. 
    
## Usage

In order to train SABD, a json have to be created with the argument values of SABD. 
Some json samples are shown in the folder "json_experiments".

Run python script experiments/sabd.py to perform the experiments.

```
#Examples
python3 experiments/sabd.py -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/sabd_eclipse_test.json "recall_rate.window=365"
python3 experiments/sabd.py -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/sabd_open_office_test.json "recall_rate.window=365"
python3 experiments/sabd.py -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/sabd_netbeans_test.json "recall_rate.window=365"
python3 experiments/sabd.py -F HOME_DIR/experiments with HOME_DIR/duplicate-bug-report/json_parameters/sabd_mozilla_test.json "recall_rate.window=365"
```


# Citation
The paper was accepted and will be published in MSR 2020.

If the code helps your research, please consider to cite our work:

    A Soft Alignment Model for Bug Deduplication. 
    Irving Muller Rodrigues, Daniel Aloise, Eraldo Rezende Fernandes, and Michel Dagenais.
    In 17th International Conference on Mining Software Repositories (MSR ’20), October 5–6, 2020, Seoul, Republic of Korea.
    ACM, New York, NY, USA, 12 pages.
    https://doi.org/10.1145/3379597.3387470
    
    @INPROCEEDINGS {rodrigues2020,
        author    = "Irving Muller Rodrigues, Daniel Aloise, Eraldo Rezende Fernandes, and Michel Dagenais",
        title     = "A Soft Alignment Model for Bug Deduplication",
        booktitle = "17th International Conference on Mining Software Repositories (MSR ’20)",
        year      = "2020",
        address   = "Seoul, Republic of Korea",
        month     = "oct",
        publisher = "ACM"
        note      = "forthcoming"
    }

# License
```
MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
