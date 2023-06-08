from bidict import bidict
import pandas as pd
import random
import os
from Datasets import PickleMappingsDataset
from Datasets import KBLRNLiteralsDataset
import pickle
from tqdm import tqdm
import math


class FB15KRandomPick:
    def __init__(self, path: str) -> None:
        self.dataset_names = ("train", "valid", "test")
        self.path = path
        self._inputDataset = PickleMappingsDataset(path, False)
        self._inputDataset.load_data()
        

    def numeric_to_half(self, numeric_file_path, new_ent2id, all_dfs):
        filename = "FB15K_NumericalTriples.txt"

        new_entities = []
        new_datas = []
        _literalsDataset = KBLRNLiteralsDataset(numeric_file_path)
        _literalsDataset.load_data(self._inputDataset.ent2id)
        
        
        datas = pd.read_csv(os.path.join(numeric_file_path,filename), header=None)
        test_df, train_df, valid_df = all_dfs

        # for index, _ in tqdm(datas.iterrows()):
        for tupel in tqdm(_literalsDataset.triples['train']):
            # row = datas.loc[index].values
            # tmp_arr = row[0].split("\t")

            # ent = tmp_arr[0]
            # take a look
            # ent_id = self._inputDataset.ent2id[ent]
            ent_id = tupel[0]
            
            if (
                ent_id in set(test_df.iloc[:, 0])
                and ent_id in set(test_df.iloc[:, 2])
                and ent_id in set(train_df.iloc[:, 0])
                and ent_id in set(train_df.iloc[:, 2])
                and ent_id in set(valid_df.iloc[:, 0])
                and ent_id in set(valid_df.iloc[:, 2])
            ):
              new_entities.append(ent_id)
            # if ent in new_ent2id.keys():
            #   new_datas.append(tmp_arr)

        for index, _ in tqdm(datas.iterrows()):
          row = datas.loc[index].values
          tmp_arr = row[0].split("\t")

          ent = tmp_arr[0]
          # take a look
          
          if ent not in self._inputDataset.ent2id.keys():
            continue
          
          ent_id = self._inputDataset.ent2id[ent]
          
          
          if ent_id in new_entities:
            new_datas.append(tmp_arr)
      
        final_df = pd.DataFrame(new_datas)
        final_df.to_csv(filename, sep="\t", index=False, header=None)
        print(final_df)

    def relation_to_half(self):

        # if 'relational' in self.path:
        #   for name in self.dataset_names:
        #     if name in self.path:
        #       filename = name

        files = os.listdir(self.path)

        # randomly pick entities from ent2id.pkl
        new_ent2id = bidict()
        new_datas = []
        new_datas_rows = 0
        all_dfs = []

        for k in self._inputDataset.ent2id:
            if new_datas_rows > math.floor(len(self._inputDataset.ent2id.keys()) * 0.5):
                break
            random_num = random.randint(0, 9)
            if random_num > 5:
                new_ent2id[k] = self._inputDataset.ent2id[k]

                new_datas_rows += 1

        # with open('new_ent2id.pkl', 'wb') as handle:
        #   pickle.dump(new_ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)

        for file in files:
            filename = ""
            if file.endswith(".txt"):
                filename = file[: file.find(".")]

                read_path = os.path.join(self.path, file)
                datas = pd.read_csv(read_path, header=None)

                # print(len(self._inputDataset.ent2id.keys()))
                # print(new_bidict)

                for index, _ in tqdm(datas.iterrows()):
                    row = datas.loc[index].values

                    tmp_arr = [int(x) for x in row[0].split("\t")]

                    if (
                        tmp_arr[0] in new_ent2id.inverse
                        and tmp_arr[2] in new_ent2id.inverse
                    ):
                        new_datas.append(tmp_arr)

                final_df = pd.DataFrame(new_datas)
                all_dfs.append(final_df)
                final_df.to_csv(f"{filename}.txt", sep="\t", index=False, header=None)

                print(new_ent2id)
                print(len(new_ent2id))

        return new_ent2id, all_dfs


test = FB15KRandomPick("data/relational/FB15K-237")
new_ent2id, all_dfs = test.relation_to_half()
test.numeric_to_half(
    "data/numeric/KBLRN/", new_ent2id, all_dfs
)
