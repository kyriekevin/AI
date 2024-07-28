# Chapter 1

## Summary

- LLMs have transformed the field of natural language processing, which previously mostly relied on explicit rule-based systems and simpler statistical methods. The advent of LLMs introduced new deep learning- driven approaches that led to advancements in understanding, generating, and translating human language.
- Modern LLMs are trained in two main stages:

  1. **Pretrain:** The model is trained on a large corpus of unlabeled text by using the prediction of the next word as the target.
  2. **Finetune:** The model is finetuned on a smaller, labeled dataset to follow instructions.

- LLMs are based on the transformer architecture.

  1. key idea: An attention mechanism that gives the LLM selective access to the whole input sequence when generating the next word.
  2. The original transformer architecture consists of an encoder for parsing the input sequence and a decoder for generating the output sequence.
  3. LLMs for generating text and following instructions only implement decoder modules.

- Large datasets are essential for training LLMs.
  1. While the general pretraining task for GPT-like models is to predict the next word, these LLMs exhibit "emergent" properties such as the ability to generate coherent text and follow instructions.
  2. Once an LLM is pretrained, the resulting base model can be finetuned more efficiently for various downstream tasks.
  3. LLMs finetuned on custom datasets can outperform general LLMs on specific tasks.
