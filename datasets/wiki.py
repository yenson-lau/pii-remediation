import datasets
import json
import numpy as np
import pandas as pd
import os
import re
import time
from nltk.tokenize import sent_tokenize
from omegaconf import OmegaConf, DictConfig
from typing import Union

_DS_DIR = os.path.dirname(os.path.realpath(__file__))
_DS_CONF = OmegaConf.load(os.path.join(_DS_DIR, "config.yaml"))
_DATA_DIR = os.path.join(_DS_DIR, _DS_CONF.data_dir)

class WikiDatasetBuilder:
    def __init__(self, **config):
        self.config: DictConfig = OmegaConf.merge(_DS_CONF.WikiDatasetBuilder,
                                                  OmegaConf.create(config))

    def build_dataset(self) -> datasets.Dataset:
        self.wiki = datasets.load_dataset("wikipedia", self.config.hf_config, split="train")
        n_wiki = len(self.wiki)
        self.dataset_dict = dict()

        if self.config.build.shuffle:
            np.random.seed(self.config.build.random_seed)
            idx_queue = np.random.permutation(n_wiki)
        else:
            idx_queue = np.arange(n_wiki)

        self.dataset_dict["config"]: dict = OmegaConf.to_object(self.config)
        self.dataset_dict["article_titles"]: dict[int, str] = dict()

        idx_cursor = 0

        for split, lim_mb in self.config.specs.split_size_mb.items():
            self.dataset_dict[split] = []
            size_mb = 0
            start_time = time.time()

            while (size_mb < lim_mb) and (idx_cursor < n_wiki):
                article = self.wiki[int(idx_queue[idx_cursor])]
                idx_cursor += 1

                article_id = int(article["id"])
                self.dataset_dict["article_titles"][article_id] = article["title"]

                # split by one or more newlines, strip any spaces
                article_lines = re.findall(r"([^\n\s][^\n]+[^\n\s])", article["text"])

                # sentence tokenize each line, then remove any sentence with too few spaces
                article_sents = [s for s in sum(map(sent_tokenize, article_lines), [])
                                 if s.count(" ") > self.config.specs.sentence_min_spaces]

                # make to use a separate variable for each step or this will take forever!
                self.dataset_dict[split] += [dict(article_id=article_id, sentence=s)
                                             for s in article_sents]

                size_mb += sum(map(len, article_sents)) / 1024**2
                elapsed_time = time.time() - start_time

                print("Building {} split: {}/{}mb ({:3d}%, {:d}s, {:.1f}mb/s)".format(
                    split,
                    (r"{:%sd}" % len(str(lim_mb))).format(int(size_mb)),
                    lim_mb,
                    int(size_mb/lim_mb*100),
                    int(elapsed_time),
                    size_mb/elapsed_time
                ), end="\r")
            print()

        self.save_dataset_dict()
        return self.load_dataset()

    def save_dataset_dict(self, path:Union[str,None]=None):
        if path is None:
            path = os.path.join(*self._resolve_dataset_path())

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'wt', encoding='UTF-8') as f:
            json.dump(self.dataset_dict, f, indent=4)

    def load_dataset(self, path:Union[str,None]=None, split:str="train") -> datasets.Dataset:
        if path is None:
            dir, path = self._resolve_dataset_path()

        return datasets.load_dataset("json", data_files=os.path.join(dir, path), field=split)

    def _resolve_dataset_path(self) -> str:
        return os.path.join(_DATA_DIR, self.__class__.__name__), self.config.build.filename

if __name__ == "__main__":
    import sys

    if (len(sys.argv) > 1) and (sys.argv[1]=="build"):
        WikiDatasetBuilder().build_dataset()
