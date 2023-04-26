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

## Query answering results(Data is not up to date)
Query answering results with different attribute embedding models for multihop entity queries without literals. Results were computed for test queries over the FB15k-237 dataset and evaluated in terms of mean reciprocal rank (MRR) and Hits@k for k ∈ {1, 3, 10}.
| Method   | Average | 1p      | 2p      | 3p      | 2i      | 3i      | ip      | pi      | 2u      | up      |
| :-------- | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|          |         |         |         |         |   MRR   |         |         |         |         |         |
| CQD      |    0.2950     | 0.4547  | 0.2746  | **0.1997**  | 0.3393  | 0.4594  | 0.1880  | 0.2666  | **0.2601**  | **0.2127**  |
|  LitCQD |**0.2981**| **0.4553**  | **0.2814**  | 0.1974  | **0.3476**  | **0.4639**  | **0.1923**  | **0.2744**  | 0.2600  | 0.2109  |
| Query2Box| 0.2104| 0.3985 | 0.1910 | 0.1346 | 0.2379 | 0.3247 | 0.1569 | 0.1036 | 0.1904 | 0.1562 |
|          |         |         |         |         | HITS@1  |         |         |         |         |         |
| CQD  |  0.2115  | **0.3555**  | 0.1981  | **0.1421**  | 0.2346  | 0.3583  | 0.1280  | 0.1863  | **0.1648**  | **0.1359**  |
|  LitCQD |**0.2129**| 0.3539  | **0.2025**  | 0.1379  | **0.2432**  | **0.3627**  | **0.1296**  | **0.1939**  | 0.1632  | 0.1296  |
| Query2Box| 0.1217 | 0.2878| 0.1144| 0.0754| 0.1217| 0.1921| 0.0810| 0.0543| 0.0838| 0.0847|
|          |         |         |         |         | HITS@3  |         |         |         |         |         |
| CQD      |    0.3218   | 0.4987  | 0.2971  | 0.2090  | 0.3819  | 0.5075  | 0.1946  | 0.2902  | **0.2875**  | 0.2300  |
|  LitCQD |**0.3271**| **0.5021**  | **0.3057**  | **0.2093**  | **0.3907**  | **0.5149**  | **0.2050**  | **0.2982**  | 0.2872  | **0.2310**  |
| Query2Box| 0.2378| 0.4485| 0.2039| 0.1395| 0.2854| 0.3938| 0.1763| 0.1040| 0.2242| 0.1642|
|          |         |         |         |         | HITS@10 |         |         |         |         |         |
| CQD |  0.4620 | 0.6562  | 0.4225  | 0.3109  | 0.5486  | 0.6572  | 0.3074  | 0.4243  | 0.4629  | 0.3683  |
|  LitCQD |**0.4700**| **0.6574**  | **0.4446**  | **0.3199** | **0.5568** | **0.6602**  | **0.3150**  | **0.4325**  | **0.4659**  | **0.3779**  |
| Query2Box| 0.3861| 0.6188| 0.3452| 0.2552| 0.4694| 0.5772| 0.2995| 0.1973| 0.4107| 0.3015|



Query answering results for multihop entity queries with literals. Our best-performing model Complex-N3 + Attributes (KBLRN) is compared to variations thereof. Results were computed for test queries over the FB15k-237 dataset and evaluated in terms of Hit@10.
| methods                     | ai-lt  | ai-eq  | ai-gt  | 2ai    | aip    | pai    | au     |
|:---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| LitCQD                      | 0.4971 | 0.3420 | 0.3874 | 0.3581 | 0.1863 | 0.5059 | 0.3216 |
| No Value Prediction        | 0.3120 | 0.0042 | 0.2318 | 0.1580 | 0.1270 | 0.4339 | 0.0537 |
| No Attribute Exists Check  | 0.1540 | 0.1532 | 0.1737 | 0.1175 | 0.1821 | 0.4894 | 0.0309 |
| Neither of them             | 0.0015 | 0.0000 | 0.0000 | 0.0000 | 0.0943 | 0.4324 | 0.0016 |
|without attribute-specific standard deviation | 0.4928 | 0.3420 | 0.3454 | 0.3557 | 0.1914 | 0.5057 | 0.2699 |


Query answering results for multihop literal queries for test queries over the FB15k-237 dataset evaluated in terms of mean absolute error (MAE) and mean squared error (MSE).
| methods        | 1ap MAE | 1ap MSE | 2ap MAE | 2ap MSE | 3ap MAE | 3ap MSE |
|:---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| LitCQD         | 0.0493  | 0.0105  | 0.0332  | 0.0039  | 0.0407  | 0.0063  |
| Query2Box      | 0.0648  | 0.0151  | 0.0476  | 0.0067  | 0.0558  | 0.0139  |
| Mean Predictor | 0.3419  | 0.143   | 0.3461  | 0.1412  | 0.3621  | 0.1517  |




