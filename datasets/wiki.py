import gzip
import json
import numpy as np
import pandas as pd
import os
import re
import time
from datasets import Dataset, load_dataset
from nltk.tokenize import sent_tokenize
from omegaconf import OmegaConf, DictConfig
from typing import Union

_DS_DIR = os.path.dirname(os.path.realpath(__file__))
_DS_CONF = OmegaConf.load(os.path.join(_DS_DIR, "config.yaml"))
_DATA_DIR = os.path.join(_DS_DIR, _DS_CONF.data_dir)

class WikiDatasetBuilder:
    config = _DS_CONF.WikiDatasetBuilder

    def __init__(self, config:Union[dict, str, DictConfig]=dict()):
        self.config: DictConfig = OmegaConf.merge(WikiDatasetBuilder.config,
                                                  OmegaConf.create(config))

    def build_dataset(self) -> Dataset:
        self.wiki = load_dataset("wikipedia", self.config.hf_config, split="train")
        n_wiki = len(self.wiki)
        self.dataset_dict = dict()

        if self.config.build.shuffle:
            np.random.seed(self.config.build.random_seed)
            idx_queue = np.random.permutation(n_wiki)
        else:
            idx_queue = np.arange(n_wiki)

        self.dataset_dict["config"]: dict = OmegaConf.to_object(self.config)
        self.dataset_dict["article_title"]: dict[int, str] = dict()
        self.dataset_dict["article_to_dataset_id"]: dict[int, int] = dict()

        sent_min_spaces = self.config.specs.sentence_min_spaces
        sent_max_spaces = self.config.specs.sentence_max_spaces

        if sent_min_spaces is None:  sent_min_spaces = float("-inf")
        if sent_max_spaces is None:  sent_max_spaces = float("inf")

        idx_cursor = 0

        for split, lim_mb in self.config.specs.split_size_mb.items():
            self.dataset_dict[split] = []
            size_mb = 0
            start_time = time.time()

            while (size_mb < lim_mb) and (idx_cursor < n_wiki):
                # use a separate variable for each step or this will take forever!
                # also preload config variables for intense tasks

                dataset_id = int(idx_queue[idx_cursor])

                article = self.wiki[dataset_id]
                article_id = int(article["id"])

                self.dataset_dict["article_title"][article_id] = article["title"]
                self.dataset_dict["article_to_dataset_id"][article_id] = dataset_id

                idx_cursor += 1

                article_sents = [
                    s for s in WikiDatasetBuilder.text_to_sents(article["text"])
                    if  sent_min_spaces < s.count(" ") < sent_max_spaces
                ]

                self.dataset_dict[split] += [dict(article_id=article_id, sentence=s)
                                             for s in article_sents]

                size_mb += sum(map(len, article_sents)) / 1024**2
                elapsed_time = time.time() - start_time

                print("Building {} split: {:3d}% [{}/{}mb, {:d}s, {:.1f}mb/s]".format(
                    split,
                    int(size_mb/lim_mb*100),
                    (r"{:%sd}" % len(str(lim_mb))).format(int(size_mb)),
                    lim_mb,
                    int(elapsed_time),
                    size_mb/elapsed_time
                ), end="\r")
            print()

        data_file = WikiDatasetBuilder._resolve_data_file(self.config)
        self.save_dataset_dict(data_file)
        return WikiDatasetBuilder.load_dataset(data_file)

    def text_to_sents(text):
        # remove non-ascii characters
        # replace multiple consecutive spaces with a single space
        text = re.sub(r" [ ]+", " ", text.encode("ascii", "ignore").decode())

        # split by one or more newlines
        # strip any spaces
        lines = re.findall(r"([^\|\s][^\|\n]+[^\|\s])", text)

        # sentence tokenize each line and collect into a list
        sents = sum(map(sent_tokenize, lines), [])

        return sents

    def save_dataset_dict(self, data_file:Union[str,None]=None):
        if data_file is None:
            data_file = self._resolve_data_file(self.config)

        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        file = (gzip.open(data_file, 'wt', encoding='UTF-8') if data_file.endswith(".gz")
                else open(data_file, 'wt', encoding='UTF-8'))
        json.dump(self.dataset_dict, file, indent=4)
        file.close()

    def load_dataset(data_file:Union[str,None]=None, split:str="train") -> Dataset:
        if data_file is None:
            data_file = WikiDatasetBuilder._resolve_data_file()
        return load_dataset("json", data_files=data_file, field=split)

    def _resolve_data_file(config:Union[DictConfig,None]=None) -> str:
        if config is None:
            config = WikiDatasetBuilder.config
        return os.path.join(_DATA_DIR, "wiki", config.build.filename)


if __name__ == "__main__":
    import sys

    # TODO: Expand args to accept different configs / use argparse
    if (len(sys.argv) > 1):

        if(sys.argv[1]=="build"):
            WikiDatasetBuilder().build_dataset()

        elif(sys.argv[1]=="test"):
            test_config = """
            specs:
                split_size_mb:
                    train: 10
                    test: 1

            build:
                filename: 20220301.en.test.json
            """
            WikiDatasetBuilder(test_config).build_dataset()
