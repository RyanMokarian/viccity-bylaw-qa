# qa_engine.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


@dataclass
class QAConfig:
    max_seq_len: int = 384
    doc_stride: int = 128
    max_answer_len: int = 32
    use_cuda: bool = True
    model_name: str = "distilbert-base-uncased-distilled-squad"
    # Kept for compatibility with bylaw_qa_extract.py's TF-IDF prefiltering
    top_k_chunks: int = 6


class QASystem:
    def __init__(self, model_name: str = "distilbert-base-uncased-distilled-squad", use_cuda: bool = True):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.mdl = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
        self.mdl.to(self.device)
        self.mdl.eval()

    @torch.inference_mode()
    def answer_span(self, question: str, context: str, cfg: QAConfig) -> Dict[str, Any]:
        """
        Return best span answer within context for a question.

        Output:
            {
              "answer": str,
              "score": float,
              "start_char": Optional[int],
              "end_char": Optional[int]
            }
        """
        # Use overflow windows with stride for long contexts
        enc = self.tok(
            question,
            context,
            truncation="only_second",
            max_length=cfg.max_seq_len,
            stride=cfg.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            add_special_tokens=True,
            return_tensors=None,  # lists are easier to loop over per window
        )

        input_ids_list      = enc["input_ids"]
        attention_mask_list = enc.get("attention_mask", [None] * len(input_ids_list))
        token_type_ids_list = enc.get("token_type_ids", [None] * len(input_ids_list))
        offsets_list        = enc["offset_mapping"]

        best = {"answer": "", "score": 0.0, "start_char": None, "end_char": None}

        for k in range(len(input_ids_list)):
            input_ids = torch.tensor([input_ids_list[k]], dtype=torch.long, device=self.device)
            mdl_inputs = {"input_ids": input_ids}

            if attention_mask_list[k] is not None:
                mdl_inputs["attention_mask"] = torch.tensor([attention_mask_list[k]], dtype=torch.long, device=self.device)
            if token_type_ids_list[k] is not None:
                # Harmless if model ignores it
                mdl_inputs["token_type_ids"] = torch.tensor([token_type_ids_list[k]], dtype=torch.long, device=self.device)

            outputs = self.mdl(**mdl_inputs)
            start_logits = outputs.start_logits[0]  # (seq_len,)
            end_logits   = outputs.end_logits[0]    # (seq_len,)

            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs   = torch.softmax(end_logits, dim=-1)

            offsets = offsets_list[k]  # list[(start_char, end_char)] per token in this window

            window_best = self._best_span_from_probs(
                start_probs, end_probs, offsets, max_answer_len=cfg.max_answer_len
            )
            if window_best["score"] > best["score"]:
                best = window_best

        if best["start_char"] is not None and best["end_char"] is not None:
            best["answer"] = context[best["start_char"]:best["end_char"]].strip()
        else:
            best["answer"] = ""

        return best

    def _best_span_from_probs(
        self,
        start_probs: torch.Tensor,
        end_probs: torch.Tensor,
        offsets: Any,
        max_answer_len: int = 32
    ) -> Dict[str, Optional[Any]]:
        """
        Choose (start,end) maximizing start_prob * end_prob with constraints:
          - valid char offsets (non-special tokens)
          - end >= start
          - span length <= max_answer_len
        """
        seq_len = start_probs.shape[0]
        best_score = 0.0
        best_i = None
        best_j = None

        for i in range(seq_len):
            si, ei = offsets[i]
            if si == 0 and ei == 0:
                continue  # special token
            s_prob = float(start_probs[i])

            j_max = min(seq_len - 1, i + max_answer_len - 1)
            for j in range(i, j_max + 1):
                sj, ej = offsets[j]
                if sj == 0 and ej == 0:
                    continue  # special token
                # ej > si is implied for normal tokens with i<=j, but keep guard
                e_prob = float(end_probs[j])
                score = s_prob * e_prob
                if score > best_score:
                    best_score = score
                    best_i, best_j = i, j

        if best_i is None or best_j is None:
            return {"answer": "", "score": 0.0, "start_char": None, "end_char": None}

        start_char = int(offsets[best_i][0])
        end_char   = int(offsets[best_j][1])

        return {
            "answer": None,  # filled by caller after slicing
            "score": float(best_score),
            "start_char": start_char,
            "end_char": end_char
        }
