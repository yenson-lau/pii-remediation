#!/bin/sh

git submodule add https://github.com/yenson-lau/muliwai
git submodule update --init

pip install https://github.com/kpu/kenlm/archive/master.zip
pip install spacy==3.1.0 dateparser python-stdnum protobuf==3.20.1 cdifflib datasets langid faker sentencepiece fsspec tqdm sentence-transformers nltk tokenizers==0.11.3
pip install --upgrade transformers
python -m nltk.downloader punkt wordnet
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download ca_core_news_sm
python -m spacy download pt_core_news_sm
python -m spacy download zh_core_web_sm

