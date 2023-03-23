# Multi-Hop Reasoning in Knowledge Graphs with Literals

## Installation
```
git clone https://github.com/dice-group/LitCQD
conda create -n litcqd python=3.8 && conda activate litcqd
pip3 install torch==1.9.0 --find-links https://download.pytorch.org/whl/torch_stable.html
pip3 install bidict==0.21.3  gensim==4.1.2
pip3 install ray[tune]==1.9.1 simple-parsing==0.0.17 tqdm==4.62.0
pip3 install tensorboardX==2.4.1 tensorboard==2.7.0 protobuf==3.20.3
```

## Datasets and Pre-trained Models

```
wget https://hobbitdata.informatik.uni-leipzig.de/LitCQD/checkpoints_FB15K-237.zip
unzip checkpoints_FB15K-237.zip
wget https://hobbitdata.informatik.uni-leipzig.de/LitCQD/data.zip
unzip data.zip
unzip checkpoints_FB15K-237.zip
```

- `data/FB15k-237-q2b` contains the FB15K-237 dataset including generated queries provided by the authors of the Query2Box paper.
They can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).
- `data/scripts/generated`: Generated queries for different datasets used in this thesis.
- `data/scripts/data`:
    - `numeric`: The numeric attribute data from different sources.
    - `relational`: The relational data for the FB15K-237 and LitWD1K datasets.
    - `textual`: The textual descriptions and entity names form different sources.

Sources:
- `KBLRN`: https://github.com/mniepert/mmkb
- `LiteralE`: https://github.com/SmartDataAnalytics/LiteralE
- `TransEA`: https://github.com/kk0spence/TransEA
- `LitWD1K`: https://github.com/GenetAsefa/LiterallyWikidata

The FB15K-237 data provided by Query2Box is described as the following by the authors:
- `train.txt/valid.txt/test.txt`: KG edges
- `id2rel/rel2id/ent2id/id2ent.pkl`: KG entity relation dicts
- `train-queries/valid-queries/test-queries.pkl`: `defaultdict(set)`, each key represents a query structure, and the value represents the instantiated queries
- `train-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`)
- `valid-easy-answers/test-easy-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the answers obtained in the training graph (edges in `train.txt`) / valid graph (edges in `train.txt`+`valid.txt`)
- `valid-hard-answers/test-hard-answers.pkl`: `defaultdict(set)`, each key represents a query, and the value represents the **additional** answers obtained in the validation graph (edges in `train.txt`+`valid.txt`) / test graph (edges in `train.txt`+`valid.txt`+`test.txt`)


### Query Generation

To generate the complex queries in `data/scripts/generated`, the python script `data/scripts/main.py` has been used.
It requires yaml configuration files. For example, using the following configuration, the script generates queries for the FB15K-237 dataset and the attribute and descriptions data provided by LiteralE:
```yaml
input: 
  name: PickleMappings # rel2id/ent2id mappings are stored as pickle files
  header: False # triples csv file does not have a header line
  path: data/relational/FB15K-237
  has_inverse: True # contains reciprocal relations
  add_inverse: False # add reciprocal relations
  add_attr_exists_rel: False # add facts representing the existence of an attribute
literals:
  name: literale
  path: data/numeric/LiteralE
  normalize: True # apply min-max normalization
  valid_size: 1000
  test_size: 1000
descriptions:
  path: data/textual/LiteralE
  name_path: data/textual/DKRL
  google_news: True # use word embeddings based on Google News or use self-trained
  jointly: False # Generated queries for joint learning
  valid_size: 1000
  test_size: 1000
output:
  name: file # store output as csv files
  path: test_output
  queries:
    generate: True # generate complex queries
    path: test_output
    print_debug: True
    complex_train_queries: False # generate complex queries for training data; required by Query2Box
```
Which configuration file to use needs to be changed in `data/scripts/main.py`.

### Checkpoints

`checkpoints_FB15K-237/checkpoint_orig_cqd_official` contains the pre-trained CQD model made publicly available as described [here](https://github.com/pminervini/KGReasoning/blob/main/CQD.md).

`checkpoints_FB15K-237` contains checkpoints for the CQD and Query2Box approaches evaluated in this thesis. The Query2Box-based approaches have a `_q2b` postfix. The approaches using description data have a `_desc` or `_desc_trained` postfix if they are not using the word embeddings based on Google News.

`checkpoints_litwd1k` contains checkpoints for the different attribute models evaluated.

## Reproducing the results in the thesis

The train scripts do not have to be run to reproduce the results as the pre-trained models are part of the repository.

### Sections 5.3 - 5.6 (Different Attribute Models)

- `attribute_prediction_train.sh`: Trains the different approaches using a grid search as described in the thesis.
- `attribute_prediction_results.sh`: Evaluate the trained models from `attribute_prediction_train.sh` by computing their achieved MRR and MAE on 1p and 1ap test queries of the LitWD1K dataset.

### Section 5.7 (CQD)
- `cqd_train.sh`: Train the approaches using CQD for the different datasets and word embeddings.
- `cqd_results.sh`: Evaluates the trained models on complex (attribute) queries and performs the other evaluations mentioned in the thesis. This includes answering the two example queries using `eval_attribute_filtering_example.py` and `eval_description_embeddings_example.py`.
- `cqd_results_other.sh`: Other evaluations not mentioned in the thesis.

### Section 5.8 (Query2Box)
- `query2box_train.sh`: Train the Query2Box-based approach with and without the attribute model.
- `query2box_results.sh`: Evaluates the trained model on complex (attribute) queries and lists box sizes of some attributes as reported in the thesis.
- `query2box_results_other.sh`: Other evaluations not mentioned in the thesis.

### Other scripts

The files in `data/scripts` like `normalize.py` and `normalize_error.py` provide further means for manual evaluation.
`attr_query_stats.py` also reports the MAE a random guesser achieves when answering the attribute value prediction queries.

### Visualization & Grid Search Results
To create the graphics in the thesis based on the results of the grid search, the Jupyer Notebook `Visualization/plot.ipynb` is used.
The csv files are a copy-paste from the console output of ray tune after finishing the grid search.

The implementation is based on the publicly available implementation of Query2Box ([Link](https://github.com/snap-stanford/KGReasoning)) .