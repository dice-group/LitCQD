import os

from Datasets import *
from Datasets import DescriptionsDataset_GoogleNews
from query_generation import QueryGenerator
from output import *
import yaml


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update({'_Config__'+k: v for k, v in entries.items()})

    def _create_folder(self, path):
        os.makedirs(path, exist_ok=True)

    @property
    def inputDataset(self):
        if not hasattr(self, '_inputDataset'):
            print(f"Loading input dataset...")
            path = self.__input["path"]
            if not os.path.isdir(path):
                raise FileNotFoundError(f"directory '{path}' does not exist")

            name = self.__input["name"]
            header = self.__input['header']
            if name.lower() == "default":
                self._inputDataset = DefaultDataset(path, header)
            elif name.lower() == "picklemappings":
                self._inputDataset = PickleMappingsDataset(path, header)
            else:
                raise NameError(f"unknown input dataset '{name}'")

            self._inputDataset.load_data()

            add_attr_exists_rel = self.__input["add_attr_exists_rel"]
            has_inverse = self.__input["has_inverse"]
            if add_attr_exists_rel:
                self._inputDataset.add_attribute_exists_relations(self.literalsDataset, has_inverse)

            add_inverse = self.__input["add_inverse"]
            if add_inverse:
                self._inputDataset.add_inverse_relations()
        return self._inputDataset

    @property
    def literalsDataset(self):
        if not hasattr(self, '_literalsDataset'):
            print(f"Loading literals dataset...")
            path = self.__literals["path"]
            if not os.path.isdir(path):
                raise FileNotFoundError(f"directory '{path}' does not exist")

            name = self.__literals['name']
            if name.lower() == "literale":
                self._literalsDataset = LiteralELiteralsDataset(path)
            elif name.lower() == "transea":
                self._literalsDataset = TransEALiteralsDataset(path)
            elif name.lower() == "kblrn":
                self._literalsDataset = KBLRNLiteralsDataset(path)
            elif name.lower() == "litwd":
                self._literalsDataset = LitWDLiteralsDataset(path)
            else:
                raise NameError(f"unknown literals dataset '{name}'")

            self._literalsDataset.load_data(self.inputDataset.ent2id)
            if self.__literals['normalize']:
                if self.__literals['std']:
                  self._literalsDataset.normalize_values(use_std=True)
                else:
                  # self._literalsDataset._z_score()
                  self._literalsDataset.normalize_values()

            
            
            valid_size = int(self.__literals["valid_size"])
            if valid_size > 0:
                self._literalsDataset.split_eval_data('valid', valid_size)

            test_size = int(self.__literals["test_size"])
            if test_size > 0:
                self._literalsDataset.split_eval_data('test', test_size)

        return self._literalsDataset

    @property
    def descriptionsDataset(self):
        if not hasattr(self, '_descriptionsDataset'):
            if '_Config__descriptions' not in self.__dict__:
                # no description input specified
                self._descriptionsDataset = DescriptionsDataset('', {"train": [], "valid": [], "test": []}, {"train": [], "valid": [], "test": []})
                return self._descriptionsDataset
            print(f"Loading descriptions dataset...")
            path = self.__descriptions["path"]
            if not os.path.isdir(path):
                raise FileNotFoundError(f"directory '{path}' does not exist")
            name_path = self.__descriptions["name_path"]
            if not os.path.isdir(name_path):
                raise FileNotFoundError(f"directory '{name_path}' does not exist")

            jointly = self.__descriptions['jointly']
            google_news = self.__descriptions['google_news']
            if jointly:
                if google_news:
                    self._descriptionsDataset = DescriptionsDatasetJointly_GoogleNews(path)
                else:
                    self._descriptionsDataset = DescriptionsDatasetJointly(path, name_path)
            else:
                if google_news:
                    self._descriptionsDataset = DescriptionsDataset_GoogleNews(path, name_path)
                else:
                    self._descriptionsDataset = DescriptionsDataset(path)
            self._descriptionsDataset.load_data(self.inputDataset.ent2id)

            valid_size = int(self.__descriptions["valid_size"])
            if valid_size > 0:
                self._descriptionsDataset.split_eval_data('valid', valid_size)

            test_size = int(self.__descriptions["test_size"])
            if test_size > 0:
                self._descriptionsDataset.split_eval_data('test', test_size)
        return self._descriptionsDataset

    @property
    def datasetWriter(self):
        if not hasattr(self, '_datasetWriter'):
            path = os.path.join('generated', self.__output["path"])
            self._create_folder(path)
            name = self.__output["name"]
            if name.lower() == "file":
                self._datasetWriter = ThesisDatasetWriter(path)
            elif name.lower() == "dummy":
                self._datasetWriter = DummyDatasetWriter(path)
            else:
                raise NameError(f"unknown output writer '{name}'")
        return self._datasetWriter

    @property
    def queryGenerator(self):
        if not hasattr(self, '_queryGenerator'):
            if not self.__output['queries']['generate']:
                self._queryGenerator = None
            else:
                path = os.path.join('generated', self.__output['queries']['path'])
                self._create_folder(path)
                print_debug = self.__output['queries']['print_debug']
                complex_train_queries = self.__output['queries']['complex_train_queries']
                add_attr_exists_rel = self.__input["add_attr_exists_rel"]
                if add_attr_exists_rel:
                    multiplier = 2 if self.__input["has_inverse"] or self.__input["add_inverse"] else 1
                    attr_exists_threshold = len(self.inputDataset.rel2id) - len(self.literalsDataset.attr2id) * multiplier
                else:
                    attr_exists_threshold = None
                desc_jointly = type(self.descriptionsDataset) == DescriptionsDatasetJointly or type(self.descriptionsDataset) == DescriptionsDatasetJointly_GoogleNews
                self._queryGenerator = QueryGenerator(
                    path,
                    print_debug=print_debug,
                    triples=self.inputDataset.triples,
                    triples_attr=self.literalsDataset.triples,
                    do_valid=True,
                    do_test=True,
                    attr_exists_threshold=attr_exists_threshold,
                    complex_train_queries=complex_train_queries,
                    descriptions=self.descriptionsDataset.triples if desc_jointly else self.descriptionsDataset.vectors,
                    desc_jointly=desc_jointly
                )
        return self._queryGenerator


configMap = yaml.safe_load(open('./config_fb15k-237.yaml', 'r'))
config = Config(**configMap)
print(config.inputDataset)
print(config.literalsDataset)
print(config.descriptionsDataset)
config.datasetWriter.write(config.inputDataset)
config.datasetWriter.write(config.literalsDataset)
config.datasetWriter.write(config.descriptionsDataset)
if config.queryGenerator:
    print("Generating queries...")
    _ = config.queryGenerator.create_queries()

print("Done.")
