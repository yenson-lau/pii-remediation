import datasets as ds
import numpy as np
import pandas as pd
import re
import time
from nltk.tokenize import sent_tokenize
from omegaconf import OmegaConf
from os import get_terminal_size, makedirs, path
from typing import Optional, Sequence
from unidecode import unidecode

__DIR__ = path.dirname(path.realpath(__file__))


# Wiki
def build_wiki(
    hf_config: str = "20220301.en",
    split_sizes_MB: dict[str, float] = {"train": 1024, "test": 256},
    shuffle: bool = True,
    random_seed: int = 0,
    sent_min_spaces: Optional[int] = 5,
    sent_max_spaces: Optional[int] = 200,
    output_dir: str = path.join(__DIR__, "..", "_data", "wiki"),
    save_compressed: bool = True,
) -> ds.Dataset:

    wiki_full = ds.load_dataset("wikipedia", hf_config, split="train")
    n_wiki_full = len(wiki_full)

    if shuffle:
        np.random.seed(random_seed)
        article_queue = np.random.permutation(n_wiki_full)
    else:
        article_queue = np.arange(n_wiki_full)

    if sent_min_spaces is None:  sent_min_spaces = float("-inf")
    if sent_max_spaces is None:  sent_max_spaces = float("inf")

    article_info = dict(id=[], title=[], dataset_id=[])
    makedirs(output_dir, exist_ok=True)

    for split, lim_mb in split_sizes_MB.items():
        data = dict(article_id=[], text=[])

        size_MB = 0
        start_time = time.time()

        def print_bar(size_MB, term_len=None):
            if term_len is None:
                try:
                    term_len = get_terminal_size().columns
                except OSError:
                    term_len = 80

            elapsed_time = time.time() - start_time
            size_mb_str = (r"{:%sd}" % len(str(lim_mb))).format(int(size_MB))
            frac = size_MB/lim_mb
            rate = size_MB/elapsed_time

            desc = f"Building {split} split: {int(frac*100):3d}% "
            stats = f" {size_mb_str}/{lim_mb}MB [{int(elapsed_time):d}s, {rate:.1f}MB/s]"

            bar_len = term_len - len(desc) - len(stats) - 2
            fill_len = int(frac * bar_len)
            bar = "|" + "█"*fill_len + " "*(bar_len-fill_len) + "|"

            print(desc+bar+stats , end="\r")

        # optimizing for loops:
        # - use a separate variable for each step
        # - preload config variables for intense tasks
        for articles_parsed, dataset_id in enumerate(map(int, article_queue)):
            if size_MB > lim_mb:
                article_queue = article_queue[articles_parsed:]
                break

            article = wiki_full[dataset_id]
            article_id = int(article["id"])

            article_info["id"].append(article_id)
            article_info["title"].append(article["title"])
            article_info["dataset_id"].append(dataset_id)

            article_sents = [
                s for s in _extract_sentences(article["text"])
                if sent_min_spaces <= s.count(" ") <= sent_max_spaces
            ]
            data["article_id"] += [article_id] * len(article_sents)
            data["text"] += article_sents

            size_MB += sum(map(len, article_sents)) / 1024**2
            print_bar(size_MB)
        print()

        data_file = path.join(output_dir, f"{split}_data.csv") + (".gz" if save_compressed else "")
        pd.DataFrame(data).to_csv(data_file, index=False)
        del data

    articles_file = path.join(output_dir, f"articles.csv") + (".gz" if save_compressed else "")
    pd.DataFrame(article_info).to_csv(articles_file, index=False)

    OmegaConf.save(
        config=OmegaConf.create(dict(
            hf_config = hf_config,
            split_sizes_MB = split_sizes_MB,
            shuffle = shuffle,
            random_seed = random_seed,
            sent_min_spaces = sent_min_spaces,
            sent_max_spaces = sent_max_spaces,
        )),
        f = path.join(output_dir, "config.yaml")
    )

    return load_wiki(directory=output_dir, splits=split_sizes_MB, force_reload=True)


# Utils
def load_wiki(directory: str, splits: Sequence[str], force_reload: bool = False) -> ds.Dataset:
    def data_file(split):
        filename =  path.join(directory, f"{split}_data.csv")
        filename += "" if path.isfile(filename) else ".gz"
        return filename

    return ds.load_dataset(
        "csv",
        data_files = {split: data_file(split) for split in splits},
        download_mode = "force_redownload" if force_reload else "reuse_dataset_if_exists"
    )

def _extract_sentences(text, use_unidecode=False):
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


# Testing
def _article_id_test(dataset: dict[str, ds.Dataset]):
    ids = None
    for data in dataset.values():
        _ids = set(data["article_id"])
        ids = _ids if ids is None else (ids & _ids)
    print(f"Intersecting article ids: {ids}")


# Main
if __name__ == "__main__":
    import sys

    # TODO: Expand args to accept different configs / use argparse
    if (len(sys.argv) > 1):
        hf_config = "20220301.en"
        output_dir = path.join(__DIR__, "..", "_data", "wiki")
        config = dict(
            hf_config = hf_config,
            shuffle= True,
            random_seed = 0,
            sent_min_spaces = 5,
            sent_max_spaces = 200,
        )

        if(sys.argv[1]=="build"):
            build_wiki(
                split_sizes_MB = {"train": 1024, "val": 128, "test": 128},
                output_dir = output_dir,
                save_compressed = True,
                **config
            )

        elif(sys.argv[1]=="test"):
            build_wiki(
                split_sizes_MB = {"train": 1, "val": .1, "test": .1},
                output_dir = output_dir + "_test",
                save_compressed = False,
                **config
            )
