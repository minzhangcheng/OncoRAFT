"""
Shared model definitions for training and inference.

MultitaskLlamaModel: Used during training.
MultitaskScoreModel: Used during inference.
"""
import re
import torch
import torch.nn.functional as F
from config import SCORE_HEAD


def build_score_head(hidden_size=None, intermediate=None, dropout=None):
    """Build the score head shared across training and inference."""
    hidden_size = hidden_size or SCORE_HEAD["hidden_size"]
    intermediate = intermediate or SCORE_HEAD["intermediate"]
    dropout = dropout or SCORE_HEAD["dropout"]
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_size, intermediate),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(intermediate, 1),
        torch.nn.Sigmoid()
    )


class MultitaskLlamaModel(torch.nn.Module):
    """Training model: base LM + LoRA + score_head, joint LM + MSE loss."""

    def __init__(self, base_model, hidden_size=None, dropout=None):
        super().__init__()
        self.base_model = base_model
        self.score_head = build_score_head(hidden_size, dropout=dropout)

    def forward(self, input_ids, attention_mask, labels=None,
                score_positions=None, true_scores=None):
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            labels=labels, output_hidden_states=True
        )
        lm_loss = outputs.loss if labels is not None else None
        hidden_states = outputs.hidden_states[-1]

        batch_size = hidden_states.size(0)
        if score_positions is not None:
            score_positions = score_positions.clamp(0, hidden_states.size(1) - 1)
            score_hidden = hidden_states[torch.arange(batch_size), score_positions]
        else:
            score_hidden = hidden_states[:, -1, :]

        predicted_scores = self.score_head(score_hidden).squeeze(-1)

        score_loss = None
        if true_scores is not None:
            true_scores = true_scores.to(predicted_scores.dtype)
            score_loss = F.mse_loss(predicted_scores, true_scores)

        total_loss = None
        if lm_loss is not None and score_loss is not None:
            total_loss = lm_loss + score_loss
        elif lm_loss is not None:
            total_loss = lm_loss
        elif score_loss is not None:
            total_loss = score_loss

        return {
            "loss": total_loss,
            "lm_loss": lm_loss,
            "score_loss": score_loss,
            "logits": outputs.logits,
            "predicted_scores": predicted_scores
        }


class MultitaskScoreModel(torch.nn.Module):
    """Inference model: base LM + score_head, score prediction + generation."""

    def __init__(self, lm_model, hidden_size=None, dropout=None):
        super().__init__()
        self.lm_model = lm_model
        self.score_head = build_score_head(hidden_size, dropout=dropout)

    def predict_score(self, input_ids, attention_mask, score_positions):
        """Extract score from score_head at the last prompt token position."""
        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            batch_size = hidden_states.size(0)
            score_positions = score_positions.clamp(0, hidden_states.size(1) - 1)
            score_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                score_positions
            ]
            scores = self.score_head(score_hidden).squeeze(-1)
        return scores.cpu().float().numpy()

    def generate(self, input_ids, attention_mask, **kwargs):
        """Proxy to lm_model.generate()."""
        return self.lm_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )


def extract_response_score(text):
    """Extract numeric score from model output text (training label parsing)."""
    answer = text.strip()
    lines = [line.strip() for line in answer.split('\n') if line.strip()]
    if not lines:
        return None

    # Direct score/likelihood match
    for line in lines:
        score_match = re.search(r'score:\s*(\d+\.\d+)', line, re.IGNORECASE)
        if score_match:
            return min(1.0, max(0.0, float(score_match.group(1))))
        likelihood_match = re.search(r'likelihood:\s*(\d+\.\d+)', line, re.IGNORECASE)
        if likelihood_match:
            return min(1.0, max(0.0, float(likelihood_match.group(1))))

    # Standalone number
    for line in lines:
        num_match = re.search(r'(?<![a-zA-Z0-9])(\d+\.\d+)(?![a-zA-Z0-9])', line)
        if num_match:
            value = float(num_match.group(1))
            if 0 <= value <= 1:
                return value

    # Keyword-based proximity search
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['score', 'likelihood', 'probability', 'prediction']):
            for j in range(i + 1, min(i + 5, len(lines))):
                if j < len(lines):
                    num_match = re.search(r'(?<![a-zA-Z0-9])(\d+\.\d+)(?![a-zA-Z0-9])', lines[j])
                    if num_match:
                        value = float(num_match.group(1))
                        if 0 <= value <= 1:
                            return value

    # Sentiment-based fallback
    relevant_text = ' '.join(lines[-15:]).lower()
    high_terms = ['high likelihood', 'highly likely', 'strong response', 'good response',
                  'very effective', 'complete response', 'positive response']
    medium_terms = ['moderate likelihood', 'partial response', 'some response',
                    'moderate response', 'may respond']
    low_terms = ['low likelihood', 'unlikely', 'poor response', 'minimal response',
                 'resistance', 'not effective', 'disease progression']
    high_count = sum(1 for term in high_terms if term in relevant_text)
    medium_count = sum(1 for term in medium_terms if term in relevant_text)
    low_count = sum(1 for term in low_terms if term in relevant_text)
    if high_count > medium_count and high_count > low_count:
        return 0.8
    elif low_count > high_count and low_count > medium_count:
        return 0.2
    else:
        return 0.5
