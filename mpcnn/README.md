# mpcnn
Tensorflow Implementation: Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks

This project implement the network (slightly different) from the paper [Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks](https://pdfs.semanticscholar.org/0f69/24633c56832b91836b69aedfd024681e427c.pdf?_ga=2.189917163.1125227972.1499913967-1950509983.1499913967)


## Test
Tested on the training set of [Quora Question Pair](https://www.kaggle.com/c/quora-question-pairs/data)

- word2vec word-embedding: 0.33621892 logloss
- glove word-embedding: 0.3132183 logloss

## Details/Problem
As the input of sentence has variable length, padding is usually used for creating tensor with same shape, which enables parallel computation and provides better computational locality. However, padding will affect the result of mean-pool and min-pool because there are lots of zeros added to the sample.

- Mean pool:
  - Problem: direct use mean operation would include the padding zero
  - Solved by: sum(output of conv) / sentence_length_tensor
- Min pool (not using in the code, to be improved):
  - Problem: min pool would always return zero due to padding zero
  - Not Exactly Same: use min(output_of_conv + min_mask)
    - Min_mask is 2d tensor. If t-th input is padding zero, then the t-th value of
      the mask is 1e7 (large value) such that the min pool value is less affected
      by padding sequence. (P.s. conv.layer using SAME padding method and
      the min pool value is not exact equal to that without padding
      sequence)

## TO BE CONTINUED
- extract and consolidate the code from the notebook into package
