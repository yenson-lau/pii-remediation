from datasets import Dataset
from omegaconf import OmegaConf
from transformers import (BertTokenizerFast,
                          BertConfig,
                          BertForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments)
from os import path

__DIR__ = path.dirname(path.realpath(__file__))


class BertPretrainer:
    def __init__(self,
        dataset: Dataset,
        text_col: str = "text",
        vocab_size = 20_000,
        base_model = "bert-base-cased",
        max_length = 128,
        tokenizer_dir = "_data/pretrain/tokenizer",
        model_dir = "_data/pretrain/model",
    ):
        self.dataset = dataset
        self.text_col = text_col
        self.vocab_size = vocab_size
        self.base_model = base_model
        self.max_length = max_length
        self.tokenizer_dir = tokenizer_dir
        self.model_dir = model_dir

        # Outputs
        self.tokenizer = None
        self.tokenized_dataset = None
        self.model = None

    def train(self,
        tokenize_params: dict = (),
        mlm_probability: float = 0.15,
        bert_config: dict = (),
        training_args: dict = (),
    ):
        self.train_tokenizer()
        self.tokenize_dataset(self, **tokenize_params)
        self.train_mlm(mlm_probability, bert_config, training_args)

    def train_tokenizer(self):
        self.tokenizer = (BertTokenizerFast
                            .from_pretrained(self.base_model)
                            .train_new_from_iterator(self.dataset[self.text_col], self.vocab_size))
        self.tokenizer.model_max_length = self.max_length

        self.tokenizer.save_pretrained(self.tokenizer_dir)

    def tokenize_dataset(self, **tokenize_params):
        _tokenize_params = dict(batched = True, num_proc = 4)
        _tokenize_params.update(tokenize_params)

        assert self.tokenizer is not None, "tokenizer needs to be trained"
        self.tokenizer.model_max_length = self.max_length

        tokenize_function = lambda ex: self.tokenizer(ex[self.text_col], truncation=True)
        self.tokenized_dataset = self.dataset.map(tokenize_function,
                                                  remove_columns = list(self.dataset.features),
                                                  **_tokenize_params)

    def train_mlm(self,
        mlm_probability: float = 0.15,
        bert_config: dict = dict(),
        training_args: dict = dict()
    ):
        assert self.tokenizer is not None, "tokenizer needs to be trained"
        assert self.tokenized_dataset is not None, "data needs to be tokenized"

        # input 1: tokenizer
        data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer,
                                                        mlm_probability = mlm_probability)

        _bert_config = dict(max_position_embeddings=self.max_length)
        _bert_config.update(bert_config)
        bert_config = BertConfig(vocab_size = self.tokenizer.vocab_size, **_bert_config)

        self.model = BertForMaskedLM(config = bert_config)

        _training_args = dict(num_train_epochs = 1,
                              per_device_train_batch_size = 128,
                              save_steps = 10_000,
                              save_total_limit = 2,
                              prediction_loss_only = True,
                              output_dir = self.model_dir,
                              overwrite_output_dir = True)
        _training_args.update(training_args)
        training_args = TrainingArguments(**_training_args)

        trainer = Trainer(model = self.model,
                          args = training_args,
                          data_collator = data_collator,
                          train_dataset = self.tokenized_dataset)

        trainer.train()
        trainer.save_model(self.model_dir)