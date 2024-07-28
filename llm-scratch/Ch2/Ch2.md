# Chapter 2

## Summary

- LLMs require textual data to be converted into numerical vectors known as embeddings. Embeddings transform discrete data into continuous vectors spaces.
  1. Raw text is broken into tokens, which can be words or characters.
     1. Special tokens can be added to enhance the model's understanding and handle various contexts.
     2. The byte pair encoding (BPE) tokenizer can efficiently handle unknown words by breaking them down into subword units or individual characters.
  2. The tokens are converted into integer representations, termed token IDs.
- We use a sliding window approach on tokenized data to generate input-target pairs for LLM training.
- Embedding layers in PyTorch function as a lookup operation, retrieving vectors corresponding to the token IDs. The resulting embedding vectors provide continuous representations of the input data.
- While token embedding provide consistent vector representations for each token, they lack a sense of the token's position within the sentence. To rectify this, two types of positional embeddings are used: 1. Absolute positional embeddings, which are fixed and do not change during training. 2. Relative positional embeddings, which are learned during training and adapt to the input sequence.
