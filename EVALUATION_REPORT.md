# Empathetic Chatbot Evaluation Report

## Model Architecture

- Transformer encoder-decoder (built from scratch)
- Embedding dimension: 512
- Attention heads: 2
- Encoder/Decoder layers: 2 each
- Dropout: 0.1
- Vocabulary size: 20195
- Total parameters: 35,412,707

## Dataset Split

- Train: 51672 samples (80%)
- Validation: 6459 samples (10%)
- Test: 6460 samples (10%)

## Training Configuration

- Optimizer: Adam (lr=1e-4, betas=(0.9, 0.98))
- Batch size: 64
- Loss: CrossEntropyLoss (ignore padding)
- Teacher forcing: Yes
- Epochs: 20
- Best model selection: Validation BLEU

## Test Set Results (Full Dataset)

### Greedy Decoding

- BLEU: 2.48
- ROUGE-L: 15.01
- chrF: 13.47
- Perplexity: 49.25

### Beam Search (width=3)

- BLEU: 2.05
- ROUGE-L: 14.44
- chrF: 11.14
- Perplexity: 49.07

## Human Evaluation

I will be giving it a 2.5/5 Score in semantic and 3.5 in syntactic.

## Qualitative Examples (Greedy)

### Example 1

**Reference:** well have you been living a happy life ? she would probably be happy about that .
**Generated:** i 'm sorry to hear that . i hope you have a good memories of her .

### Example 2

**Reference:** ah man ! you should n't feel bad about tripping . why do you feel bad about it ?
**Generated:** oh no ! did you hurt yourself ?

### Example 3

**Reference:** it is , and yeah until they are over the age of you go go every other month crazy poor babies ! !
**Generated:** she was a very good idea .

### Example 4

**Reference:** awww ... still thats upsetting
**Generated:** i 'm sorry to hear that . what 's the job is it ?

### Example 5

**Reference:** yes , but it is crazy expensive to go .
**Generated:** i 'm going to buy a lot of money .

## Qualitative Examples (Beam Search)

### Example 1

**Reference:** well have you been living a happy life ? she would probably be happy about that .
**Generated:** i 'm sorry to hear that . i hope you find someone better .

### Example 2

**Reference:** ah man ! you should n't feel bad about tripping . why do you feel bad about it ?
**Generated:** oh no ! did you get hurt ?

### Example 3

**Reference:** it is , and yeah until they are over the age of you go go every other month crazy poor babies ! !
**Generated:** yes , she had a lot of fun .

### Example 4

**Reference:** awww ... still thats upsetting
**Generated:** i 'm sorry to hear that . are you going to be a good job ?

### Example 5

**Reference:** yes , but it is crazy expensive to go .
**Generated:** i am going to take a trip .

## Implementation Details

- Multi-head attention with residual connections and layer normalization
- Sinusoidal positional encoding
- Causal masking in decoder self-attention
- Greedy and beam search decoding (beam width=3)
- Special tokens: <pad>, <bos>, <eos>, <unk>, <sep>, <emotion_X>

## Deployment

- Framework: Streamlit
- Features: Interactive chat, emotion selection, conversation history, decoding method selection
- Run: `streamlit run app.py`
