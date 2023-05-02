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

## Reproduce the results

Please refer to the `final_scripts/final_script.sh`. To config the save path of pre-trained models, please refer to `config.py` and set the corresponding variables.

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

## Query answering results
Query answering results with different attribute embedding models for multihop entity queries without literals. Results were computed for test queries over the FB15k-237 dataset and evaluated in terms of mean reciprocal rank (MRR) and Hits@k for k ∈ {1, 3, 10}.
| Method   | Average | 1p      | 2p      | 3p      | 2i      | 3i      | ip      | pi      | 2u      | up      |
| :-------- | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|          |         |         |         |         |   MRR   |         |         |         |         |         |
| CQD      |    0.295     | 0.454  | 0.275  | 0.197  | 0.339  | 0.457  | 0.188  | 0.267  | 0.261  | 0.214  |
|  LitCQD |**0.301**| **0.457**  | **0.285**  | **0.202**  | **0.350**  | **0.466**  | **0.193**  | **0.274**  | **0.266**  | **0.215**  |
| Query2Box| 0.213| 0.403 | 0.198 | 0.134 | 0.238 | 0.332 | 0.107 | 0.158 | 0.195 | 0.153 |
|          |         |         |         |         | HITS@1  |         |         |         |         |         |
| CQD  |  0.211  | 0.354  | 0.198  | 0.137  | 0.235  | 0.354  | 0.130  | 0.186  | 0.165 | **0.137**  |
|  LitCQD |**0.215**| **0.355**  | **0.206**  | **0.141**  | **0.245**  | **0.365**  | **0.129**  | **0.193**  | **0.168**  | 0.135  |
| Query2Box| 0.124 | 0.293| 0.120| 0.071| 0.124| 0.202| 0.056| 0.083| 0.094| 0.079|
|          |         |         |         |         | HITS@3  |         |         |         |         |         |
| CQD      |    0.322   | 0.498  | 0.297  | 0.208  | 0.380  | 0.508  | 0.195  | 0.290  | 0.287  | 0.230  |
|  LitCQD |**0.330**| **0.506**  | **0.309**  | **0.214**  | **0.395**  | **0.517**  | **0.204**  | **0.296**  | **0.295**  | **0.235**  |
| Query2Box| 0.240| 0.453| 0.214| 0.142| 0.277| 0.399| 0.111| 0.176| 0.226| 0.161|
|          |         |         |         |         | HITS@10 |         |         |         |         |         |
| CQD |  0.463 | 0.656  | 0.422  | 0.312  | 0.551  | 0.656  | 0.305  | 0.425  | 0.465  | 0.370  |
|  LitCQD |**0.472**| **0.660**  | **0.439**  | **0.323** | **0.561** | **0.663**  | **0.315**  | **0.434**  | **0.475**  | **0.379**  |
| Query2Box| 0.390 | 0.623| 0.356| 0.259| 0.472| 0.580| 0.203| 0.303| 0.405| 0.303|



Query answering results for multihop entity queries with literals. Our best-performing model Complex-N3 + Attributes (KBLRN) is compared to variations thereof. Results were computed for test queries over the FB15k-237 dataset and evaluated in terms of Hit@10.
| methods                     | ai-lt  | ai-eq  | ai-gt  | 2ai    | aip    | pai    | au     |
|:---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| LitCQD                      | 0.405 | 0.232 | 0.329 | 0.216 | 0.174 | 0.320 | 0.212 |
| - No Value Prediction        | 0.280 | 0.005 | 0.237 | 0.148 | 0.124 | 0.421 | 0.054 |
| - No Attribute Exists Check  | 0.203 | 0.137 | 0.128 | 0.099 | 0.156 | 0.412 | 0.002 |
| - Neither of them             | 0.002 | 0.000 | 0.000 | 0.000 | 0.086 | 0.412 | 0.002 |
|- Without attribute-specific standard deviation | 0.391 | 0.359 | 0.330 | 0.329 | 0.195 | 0.447 | 0.248 |


Query answering results for multihop literal queries for test queries over the FB15k-237 dataset evaluated in terms of mean absolute error (MAE) and mean squared error (MSE). Query2Box uses the attributes of KBLRN dataset.
| methods        | 1ap MAE | 1ap MSE | 2ap MAE | 2ap MSE | 3ap MAE | 3ap MSE |
|:---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| LitCQD         | 0.050  | 0.011  | 0.034  | 0.005  | 0.041  | 0.007  |
| Query2Box + Attribute | 0.065  | 0.015   | 0.048  | 0.007  | 0.056  | 0.014  |
| Mean Predictor | 0.341  | 0.143   | 0.346  | 0.141  | 0.362  | 0.152  |




