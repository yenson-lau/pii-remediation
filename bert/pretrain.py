import os
from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (BertConfig,
                          BertForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments)
from tokenizers import (Tokenizer,
                        models,
                        normalizers,
                        pre_tokenizers,
                        decoders,
                        trainers)
from typing import Optional, Union, Sequence

__DIR__ = os.path.dirname(os.path.realpath(__file__))
_DATA_DIR = os.path.abspath(OmegaConf.load(os.path.join(__DIR__, "config.yaml"))
                                     .data_directory.format(__dir__=__DIR__))
_PRETRAIN_DIR = os.path.join(_DATA_DIR, "_pretrain")


class WordPieceTrainer:
    def __init__(self,
        strip_accents: bool = True,
        lowercase: bool = False,
        vocab_size: int = 20_000,
        special_tokens: list[str] = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"],
    ) -> None:

        self._config = dict(vocab_size = vocab_size,
                            strip_accents = strip_accents,
                            lowercase = lowercase,
                            special_tokens = special_tokens)

        tokenizer = Tokenizer(models.WordPiece())
        tokenizer.mask_token = "[MASK]"
        tokenizer.normalizer = normalizers.BertNormalizer(strip_accents=strip_accents,
                                                          lowercase=lowercase)
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()

        self.tokenizer = tokenizer

        self.trainer = trainers.WordPieceTrainer(
            show_progress = True,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

    def train(self,
        dataset: Sequence,
        batch_size: int = 1_000
    ) -> None:

        self._config["batch_size"] = batch_size

        def batch_iterator(batch_size=batch_size):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]

        self.tokenizer.train_from_iterator(batch_iterator(),
                                           trainer=self.trainer,
                                           length=len(dataset))

    def save_tokenizer(self, path: Optional[str] = None) -> None:
        if path is None:
            os.makedirs(_PRETRAIN_DIR, exist_ok=True)
            path = os.path.join(_PRETRAIN_DIR, "tokenizer.json")
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        self.tokenizer.save(path)

    def load_tokenizer(path: Optional[str] = None) -> Tokenizer:
        if path is None:
            path = os.path.join(_PRETRAIN_DIR, "tokenizer.json")

        return Tokenizer.from_file(path)


class BertMLMPretrainer:
    def __init__(self,
        dataset: Dataset,
        tokenizer: Tokenizer,
        bert_config: dict = dict(),
        training_args: dict = dict(),
        mlm_probability: float = 0.15,
        output_dir: str = _PRETRAIN_DIR,
    ) -> None:

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.output_dir = output_dir

        _bert_config = dict()
        _bert_config.update(bert_config)
        config = BertConfig(vocab_size=len(self.tokenizer.get_vocab()), **_bert_config)

        self.model = BertForMaskedLM(config=config)

        _training_args = dict(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=64,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
        )
        _training_args.update(training_args)
        training_args = TrainingArguments(**_training_args)

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                        mlm_probability=mlm_probability)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.dataset,
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.output_dir)
