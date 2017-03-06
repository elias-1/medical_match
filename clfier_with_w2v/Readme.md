
# Welcome to [Drcubic Lab](http://www.drcubic.com/)

@(CNN-LSMT-CRFS-NER)[Drcubic|Python|Markdown]

## Steps for running

**1. run configuration**
```bash
./configure
```

**2. bazel build word2vec**
```bash
bazel build  third_party/word2vec:word2vec
```

**3. bazel build sentiemt_data_preprocess**
```bash
bazel build :prepare_ner_data
```
