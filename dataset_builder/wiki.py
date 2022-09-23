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
from typing import Optional, Union, Sequence
from unidecode import unidecode

__DIR__ = os.path.dirname(os.path.realpath(__file__))
_DS_CONF = OmegaConf.load(os.path.join(__DIR__, "config.yaml"))


class WikiDatasetBuilder:
    default_config: DictConfig = _DS_CONF.WikiDatasetBuilder

    def __init__(self,
        config: Union[dict, str, DictConfig] = dict(),
        data_file: Optional[str] = None
    ) -> None:

        self.config: DictConfig = OmegaConf.merge(WikiDatasetBuilder.default_config,
                                                  OmegaConf.create(config))

        if data_file is None:
            self.data_file = os.path.join(_DS_CONF.data_dir.format(__dir__=__DIR__),
                                          self.config.build.subdirectory,
                                          self.config.build.filename)
        else:
            self.data_file = data_file

    def build_dataset(self) -> Dataset:
        self.wiki = load_dataset("wikipedia", self.config.hf_config, split="train")
        n_wiki = len(self.wiki)
        self.dataset_dict = dict()

        if self.config.build.shuffle:
            np.random.seed(self.config.build.random_seed)
            article_queue = np.random.permutation(n_wiki)
        else:
            article_queue = np.arange(n_wiki)

        self.dataset_dict["config"]: dict = OmegaConf.to_object(self.config)
        self.dataset_dict["article_title"]: dict[int, str] = dict()
        self.dataset_dict["article_to_dataset_id"]: dict[int, int] = dict()

        sent_min_spaces = self.config.specs.sentence_min_spaces
        sent_max_spaces = self.config.specs.sentence_max_spaces

        if sent_min_spaces is None:  sent_min_spaces = float("-inf")
        if sent_max_spaces is None:  sent_max_spaces = float("inf")

        split_sizes = self.config.specs.split_size_mb

        for split, lim_mb in split_sizes.items():
            self.dataset_dict[split] = []
            size_mb = 0
            start_time = time.time()

            def print_bar(size_mb, term_len=os.get_terminal_size().columns):
                elapsed_time = time.time() - start_time
                size_mb_str = (r"{:%sd}" % len(str(lim_mb))).format(int(size_mb))
                frac = size_mb/lim_mb
                rate = size_mb/elapsed_time

                desc = f"Building {split} split: {int(frac*100):3d}% "
                stats = f" {size_mb_str}/{lim_mb}mb [{int(elapsed_time):d}s, {rate:.1f}mb/s]"

                bar_len = term_len - len(desc) - len(stats) - 2
                fill_len = int(frac * bar_len)
                bar = "|" + "█"*fill_len + " "*(bar_len-fill_len) + "|"

                print(desc+bar+stats , end="\r")

            # optimizing for loops:
            # - use a separate variable for each step
            # - preload config variables for intense tasks
            for articles_parsed, dataset_id in enumerate(map(int, article_queue)):
                if size_mb > lim_mb:
                    article_queue = article_queue[articles_parsed:]
                    break

                article = self.wiki[dataset_id]
                article_id = int(article["id"])

                self.dataset_dict["article_title"][article_id] = article["title"]
                self.dataset_dict["article_to_dataset_id"][article_id] = dataset_id

                article_sents = [
                    s for s in WikiDatasetBuilder.extract_sentences(article["text"])
                    if sent_min_spaces <= s.count(" ") <= sent_max_spaces
                ]

                self.dataset_dict[split] += [{"article_id": article_id, "text": s}
                                             for s in article_sents]

                size_mb += sum(map(len, article_sents)) / 1024**2
                print_bar(size_mb)
            print()

        self.save_dataset_dict(self.data_file)

        print("\n Running tests on data build:")
        datasets = WikiDatasetBuilder._load_test(self.data_file, split_sizes.keys())
        WikiDatasetBuilder._article_id_test(datasets)

        return datasets

    def extract_sentences(text, use_unidecode=False):
        # remove non-ascii characters
        # replace multiple consecutive spaces with a single space
        ascii_text = re.sub(r" [ ]+", " ",
                            (unidecode(text) if use_unidecode else text)
                                .encode("ascii", "ignore")
                                .decode())

        # split by one or more newlines
        # strip any spaces
        lines = re.findall(r"([^\|\s][^\|\n]+[^\|\s])", ascii_text)

        # sentence tokenize each line and collect into a list
        sents = sum(map(sent_tokenize, lines), [])

        return sents

    def save_dataset_dict(self, data_file: Optional[str] = None) -> None:
        if data_file is None:
            data_file = self.data_file

        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        file = (gzip.open(data_file, 'wt', encoding='UTF-8') if data_file.endswith(".gz")
                else open(data_file, 'wt', encoding='UTF-8'))
        json.dump(self.dataset_dict, file, indent=4)
        file.close()

    def load_dataset(
        data_file: str,
        split: str = "train",
        force_reload: bool = False
    ) -> Dataset:
        download_mode = "force_redownload" if force_reload else "reuse_dataset_if_exists"
        return load_dataset("json", data_files=data_file,
                                    field=split,
                                    download_mode=download_mode)["train"]

    def _load_test(data_file: str, splits: Sequence[str]):
        datasets = {
            split: WikiDatasetBuilder.load_dataset(data_file=data_file,
                                                   split=split,
                                                   force_reload=True)
            for split in splits
        }
        print(f"Loaded data splits:\n{datasets}")
        return datasets

    def _article_id_test(datasets: dict[str, Dataset]):
        ids = None
        for data in datasets.values():
            _ids = set(data["article_id"])
            ids = _ids if ids is None else (ids & _ids)
        print(f"Intersecting article ids: {ids}")


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
