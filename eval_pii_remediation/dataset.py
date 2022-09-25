import datasets as ds
import gzip
import json
import numpy as np
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
    text_col: str = "text",
    output_dir: str = path.join(__DIR__, "..", "_data", "wiki", "20220301.en.1gb"),
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

    for split, lim_mb in split_sizes_MB.items():
        data_dict = dict(article_title = dict(),
                         article_to_dataset_id = dict(),
                         data = [])

        size_mb = 0
        start_time = time.time()

        def print_bar(size_mb, term_len=get_terminal_size().columns):
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

            article = wiki_full[dataset_id]
            article_id = int(article["id"])

            data_dict["article_title"][article_id] = article["title"]
            data_dict["article_to_dataset_id"][article_id] = dataset_id

            article_sents = [
                s for s in _extract_sentences(article["text"])
                if sent_min_spaces <= s.count(" ") <= sent_max_spaces
            ]

            data_dict["data"] += [{"article_id": article_id, text_col: s}
                                     for s in article_sents]

            size_mb += sum(map(len, article_sents)) / 1024**2
            print_bar(size_mb)
        print()

        data_file = path.join(output_dir, f"{split}_data.json")
        if save_compressed:  data_file += ".gz"
        save_data_dict(data_dict, data_file)
        del data_dict

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

def load_wiki(directory: str, splits: Sequence[str], force_reload: bool = False) -> ds.Dataset:
    download_mode = "force_redownload" if force_reload else "reuse_dataset_if_exists"
    dataset = dict()

    for split in splits:
        data_file = path.join(directory, f"{split}_data.json")
        if not path.isfile(data_file):  data_file += ".gz"

        dataset[split] = ds.load_dataset(
            "json",
            data_files = data_file,
            field = "data",
            download_mode = download_mode)["train"]

    return dataset


# Utils
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

def save_data_dict(data_dict: dict, data_file: str) -> None:
    makedirs(path.dirname(data_file), exist_ok=True)
    file = (gzip.open(data_file, 'wt', encoding='UTF-8') if data_file.endswith(".gz")
            else open(data_file, 'wt', encoding='UTF-8'))
    json.dump(data_dict, file, indent=4)
    file.close()


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
        output_dir = path.join(__DIR__, "..", "_data", "wiki", hf_config)
        config = dict(
            hf_config = hf_config,
            shuffle= True,
            random_seed = 0,
            sent_min_spaces = 5,
            sent_max_spaces = 200
        )

        if(sys.argv[1]=="build"):
            build_wiki(
                split_sizes_MB = {"train": 1024, "test": 256},
                output_dir = output_dir + ".1gb",
                save_compressed = True,
                **config
            )

        elif(sys.argv[1]=="test"):
            build_wiki(
                split_sizes_MB = {"train": 10, "test": 1},
                output_dir = output_dir + ".test",
                save_compressed = False,
                **config
            )
