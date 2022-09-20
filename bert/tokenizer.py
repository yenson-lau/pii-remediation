import os
from datasets import Dataset
from omegaconf import OmegaConf
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from typing import Optional, Sequence, Union

__DIR__ = os.path.dirname(os.path.realpath(__file__))
_BERT_CONF = OmegaConf.load(os.path.join(__DIR__, "config.yaml"))

class WordPieceTrainer:
    directory = os.path.join(
        _BERT_CONF.data_directory.format(__dir__=__DIR__),
        _BERT_CONF.tokenizer.subdirectory,
        _BERT_CONF.tokenizer.wordpiece.subdirectory
    )
    config = _BERT_CONF.tokenizer.wordpiece

    def __init__(self,
        strip_accents: Optional[bool]  = None,
        lowercase: Optional[bool] = None,
        vocab_size: Optional[int] = None,
        special_tokens: Optional[list[str]] = None,
    ) -> None:

        if strip_accents is None:   strip_accents = WordPieceTrainer.config.strip_accents
        if lowercase is None:       lowercase = WordPieceTrainer.config.lowercase
        if vocab_size is None:      vocab_size = WordPieceTrainer.config.vocab_size

        if special_tokens is None:
            special_tokens = OmegaConf.to_object(WordPieceTrainer.config.special_tokens)

        self._config = dict(vocab_size = vocab_size,
                            strip_accents = strip_accents,
                            lowercase = lowercase,
                            special_tokens = special_tokens)

        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.normalizer = normalizers.BertNormalizer(strip_accents=strip_accents,
                                                          lowercase=lowercase)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()

        self.tokenizer = tokenizer

        self.trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    def train(self,
        dataset: Union[Dataset, Sequence],
        text_field: Optional[str] = "sentence",
        batch_size: Optional[int] = None
    ) -> None:

        if batch_size is None:
            batch_size = WordPieceTrainer.config.batch_size

        self._config["batch_size"] = batch_size

        if text_field is None:
            def batch_iterator(batch_size=batch_size):
                for i in range(0, len(dataset), batch_size):
                    yield dataset[i : i + batch_size]
        else:
            def batch_iterator(batch_size=batch_size):
                for i in range(0, len(dataset), batch_size):
                    yield dataset[i : i + batch_size][text_field]

        self.tokenizer.train_from_iterator(batch_iterator(),
                                           trainer=self.trainer,
                                           length=len(dataset))

    def save_tokenizer(self, directory: Optional[str] = None) -> None:
        if directory is None:   directory = WordPieceTrainer.directory
        if prefix is None:      prefix = WordPieceTrainer.prefix

        os.makedirs(directory, exist_ok=True)

        prefix = os.path.join(directory, prefix)
        self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        OmegaConf.save(config=OmegaConf.create({"wordpiece": self._config}),
                       f=os.path.join(directory, "config.yaml"))

    def load_tokenizer(directory: Optional[str] = None) -> Tokenizer:
        if directory is None:
            directory = os.path.join(WordPieceTrainer.directory, "tokenizer.json")

        return Tokenizer.from_file(directory)
