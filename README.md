# facebook_emotion_transformer

# Empathetic Conversational Chatbot

Transformer encoder-decoder model for empathetic dialogue generation.

## Setup

```bash
pip install torch pandas numpy nltk scikit-learn sacrebleu rouge-score streamlit
```

## Training

Run all cells in `project2_complete.ipynb`

## Inference

```bash
streamlit run app.py
```

## Files

- `project2_complete.ipynb`: Complete training pipeline
- `app.py`: Streamlit chatbot interface
- `best_model.pt`: Trained model checkpoint (I Have Manually Changed It Other For Reasons)
- `EVALUATION_REPORT.md`: Metrics and analysis

## Model Architecture

- Transformer encoder-decoder (from scratch)
- 512-dim embeddings, 2 heads, 2 layers
- Positional encoding, multi-head attention, residual connections
- Teacher forcing during training
- Greedy and beam search decoding

## Dataset

Empathetic Dialogues (Kaggle)

- Input: Emotion + Situation + Customer utterance
- Output: Agent empathetic reply
- Split: 80/10/10 train/val/test

## Results

See `EVALUATION_REPORT.md` for detailed metrics and examples.
