# Overview
Due to the scalability and complexity of the social network data, a high-qualified context for GNN prediction is hard to achieve, which significantly degrades the capacity of existing approaches in a real scenario.

We propose a solution for the aforementioned problem of fake news detection. Specifically, we denoise the large-scale data and mine critical propagation patterns, which improves the rationality and accuracy of the fake news detection process.

# Installation
```bash
pip install -r requirements.txt
```

# Datasets
Please create a new directory ```data/``` in the root directory, and download the required datasets into ```data/``` directory following the tutorial https://github.com/safe-graph/GNN-FakeNews.

# Running examples
Distributed training with multi-GPUs is available using the following command:
```bash
accelerate launch main.py --args_for_the_script
```
