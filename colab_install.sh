#!/bin/sh

# Install packages
pip install datasets evaluate transformers tokenizers
pip install "ipywidgets>=7.0,<8.0"
pip install black papermill scrapbook

# Download data
pip install b2

export B2_APPLICATION_KEY_ID=001fb03ea2a8a1c0000000001
export B2_APPLICATION_KEY=K001IiLmTwMNx5jUMJexoRI+iF1v0JQ

__DIR__=$(dirname "${BASH_SOURCE[0]}")
DATA_DIR=$__DIR__/_data
b2 sync b2://pii-project $DATA_DIR
