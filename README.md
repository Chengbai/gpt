# Learning GPT
 - Generative Pre-trained Transformer.
 - A language model trained with both supervised and reinforcement learning. 
 
# Key Concept
## tokeniser
 - tiktoken: openAI [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) tokeniser: https://github.com/openai/tiktoken
   - GPT-2: "gpt2"
   - GPT-4: "cl100k_base"
   - https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
 - sentencepiece: google's unsupervised text tokenizer and detokenizer : https://github.com/google/sentencepiece
 - string <-> int
   - string could be char, word, sub-word.
   - int: [0, vocabulary]
   - example: openAI gpt2: https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json
- viz tokenizer: https://tiktokenizer.vercel.app/?model=meta-llama%2FMeta-Llama-3-70B
- References
   - Let's build the GPT Tokenizer: https://www.youtube.com/watch?v=zduSFxRajkE&t=1711s
 
# batch/mini-batch
utilize the vector computation, optimize the CPU/GPU/Memory usage, and through put.

## context
 - option1: fixed size rolling window, partial at the beginning
 - option2: dynamic range w/ max(fixed) size - GPT
  - samples

# tensor
multi-dimentional data object.
 - shape. e.g [B, T, C]

# embedding
 - tensor look up table by row index.

# self-attention
 - attention is the communication mechanism;
 - compare with convolution nn, it preseved the `space` info, filter is sequencial moved through the `space`, and result is put back to the `same space`.
   Attention has no `space` info, have to manually encode the position info.
 - attention is within the `batch`, no cross the batches.
 - in the `encoder` block, all nodes `communicate` to each other; in `decoder` block, node will not talk to `future`. 
 - each token has an embedding, it `combines` token self-embedding + position embedding. [B, T, C].
 - key: `what do I have`. Computed via key-head([C, head_size]) applied to token embedding, [B, T, C] => [B, T, head_size]
 - query: `what am I looking for`. Computed via query-head([C, head_size]) applied to token embedding, [B, T, C] => [B, T, head_size]
 - weight: query @ key, [B, T, head_size]@[B, head_size, T].transpose(-2, -1) => [B, T, T]. 
 - value: 

# self-attention vs cross-attention
 - self-attention: **same source** are used to compute key or query or value.
 - cross-attention: **different sources** are used to compute key or query or value.

# References
- nanoGPT: https://github.com/karpathy/nanoGPT
- meta-llma: 
   - https://llama.meta.com/
   - https://github.com/meta-llama
