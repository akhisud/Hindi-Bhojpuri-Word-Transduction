# Hindi-Bhojpuri-Word-Transduction

This repo hosts the code necessary to reproduce the results of our paper **Neural machine translation based
word transduction mechanisms for Low Resource Languages** currently in review. 

***

The entire project deals with following sub-parts:

## Generating char2vec from pre-trained Hindi fastText embeddings

The pre-trained Hindi word vectors can be downloaded from [here](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md).

Running the file `generate_char2vec.py` generates the character vectors for **71 Devanagari characters** from the pre-trained word vectors.

## Models Used

We experimented with four variants of sequence-to-sequence models for our project:
- **Peeky Seq2seq Model**: Run the file `peeky_Seq2seq.py`. The implementation is based on [Sequence to Sequence Learning with Keras](https://github.com/farizrahman4u/seq2seq).

- **Alignment Model (AM)**: Run the file `attentionDecoder.py`. Following the work of _Bahdanau et al._ [1], the file `attention_decoder.py` contains the custom Keras layer based on Tensorflow backend. The original implementation can be found [here](https://github.com/datalogue/keras-attention/blob/master/models/custom_recurrents.py). A good blog post guiding the use of this implementation can be found [here](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/).
- **Heirarchical Attention Model (HAM)**: Run the file `attentionEncoder.py`. Inspired from the work of _Yang et al._ [2] Original implementation can be found [here](https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2).

- **Transformer Network**: `generate_data_for_tensor2tensor.py` generates the data as required by the Transformer network. The data is required while registering your own database (See [this](https://tensorflow.github.io/tensor2tensor/new_problem.html) for further reading). For a detailed look at installation and usage, visit their official github [page](https://github.com/tensorflow/tensor2tensor).

## Evaluation metrics
- `bleu_score.py` measures the BLEU score between the transduced and the actual Bhojpuri words averaged over the entire output file.

- `word_accuracy.py` simply measures the proportion of correctly transduced words in the output file.

***
# References 
[1] Bahdanau, D., Bengio, Y., & Cho, K. (2014). Neural Machine Translation by Jointly Learning to Align and Translate. CoRR, abs/1409.0473.

[2] Dyer, C., He, X., Hovy, E.H., Smola, A.J., Yang, Z., & Yang, D. (2016). Hierarchical Attention Networks for Document Classification. HLT-NAACL.

