from .config import Config
from .dataset import build_wiki
from .pretrain import BertPretrainer
from os import path
from typing import Optional


def run_pipeline(config):
    # Dataset
    dataset: str = config.pretrain.dataset
    dataset_config = config.dataset[dataset]

    if not path.isdir(dataset_config.output_dir):
        build_wiki(**dataset_config)

    # Pretrain
    pretrain_config = config.pretrain

    trainer = BertPretrainer(dataset=dataset_config.output_dir,
                             text_col=dataset_config.text_col,
                             vocab_size=pretrain_config.vocab_size,
                             base_model=pretrain_config.base_model,
                             max_length=pretrain_config.max_length,
                             tokenizer_dir=pretrain_config.tokenizer_dir,
                             model_dir=pretrain_config.model_dir)

    trainer.train(tokenize_params=pretrain_config.tokenize_params,
                  mlm_probability=pretrain_config.mlm_probability,
                  bert_config=pretrain_config.bert_config,
                  training_args=pretrain_config.training_args)

def main(config: Optional[str] = None):
    Config.reset_config(config)     # fresh config

    # ... make / loop over modifications
    Config.resolve_config()         # resolve the config

    run_pipeline(Config.config)


if __name__ == "__main__":
    main()
