"""Class discovery with Qwen3-VL, run in-process instead of over HTTP.

`classic_vllm.py` talks to a vLLM server on localhost:8000. vLLM is not
installed on the offline kernel and installing it there is a project of its
own, while `transformers` is already present and 102 GB of VRAM is plenty for
an 8B model. So the same prompt is issued directly to the model.

The prompt below is the paper's, kept verbatim — the label vocabulary it
produces feeds SAM3 and then the 3D lifting, so drifting from it would change
what the pipeline detects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

DEFAULT_MODEL = "/tmp/qwen3vl8b"
MAX_NEW_TOKENS = 400

# Verbatim from holi-spatial/classic_vllm.py build_messages().
PROMPT_TEMPLATE = (
    "You are labeling objects in a video frame. Identify ALL visible objects in the image. "
    "Use concise, lowercase English nouns as object class names. Be instance-aware when deciding what is present, "
    "but output only class names (no instance IDs) and list each class at most once.\n\n"
    "Granularity rules (avoid over-fragmentation):\n"
    "- Use whole-object categories at a practical level (e.g., sofa, pillow, bed, shoe, broom).\n"
    "- Do NOT split a single object into parts or components (e.g., do not use 'broom handle', 'broom head'; use 'broom').\n"
    "- Prefer common household object categories; avoid overly fine-grained subtypes.\n"
    "- Use the base class noun only: do NOT add quantity, shape, material, state, or descriptive modifiers "
    "(e.g., use 'paper' not 'stack of papers'/'a stack of paper'; use 'chair' not 'office chair'; use 'box' not 'cardboard box').\n\n"
    "No skipping rule:\n"
    "- You MUST label every distinct object you can see.\n"
    "- If you are unsure of an exact subtype, choose the closest reasonable, still-concrete supercategory (e.g., 'chair' instead of a specific chair type; "
    "'bottle' instead of a specific brand; 'bag' instead of an uncertain bag type). Do not omit objects.\n"
    "- Do NOT use non-informative words like 'object', 'thing', 'stuff', 'unknown'.\n\n"
    "Consistency with previous frames (canonical vocabulary):\n"
    "- Existing labels from previous frames: {previous}\n"
    "- If any object in the current image matches a previous label in meaning (including synonyms, singular/plural, or minor wording differences), "
    "you MUST reuse the previous label exactly and MUST NOT introduce a new synonym.\n"
    "- Only create a new label if no existing label can reasonably cover the same object meaning.\n\n"
    "Output format:\n"
    "- Return a single line of lowercase, comma-separated class names only.\n"
    "- Include classes that appeared in previous frames if they also appear here.\n"
    "- No explanations, no extra text."
)

# Words the prompt forbids; the model still emits them occasionally.
BANNED = frozenset({"object", "thing", "stuff", "unknown", "none", ""})


def consolidate(label: str, vocabulary: Sequence[str]) -> str:
    """Collapse a label onto an established base class, per the prompt's own rule.

    The prompt tells the model to use the base noun with no material, shape or
    state modifiers, and to reuse an existing label whenever the meaning
    matches. On a workshop scene it obeyed for 140 frames and then drifted:
    ``wood`` spawned ``wood panel``, ``wood composite board``, ``wood
    particleboard``, ``wood sawdust`` and twenty-two more. Each variant then
    entered the carried-forward vocabulary, where the "reuse previous labels"
    instruction *reinforced* it — 26 of 58 classes ended up as elaborations of
    one material.

    Left alone that reaches SAM3 as twenty-six near-identical text prompts over
    the same pile of lumber, and lift3d as twenty-six overlapping boxes.

    So when a label begins with a vocabulary term already established, it is a
    specialisation of that term and is folded back into it. This enforces the
    instruction rather than departing from it.
    """
    for base in vocabulary:
        if label != base and label.startswith(base + " "):
            return base
    return label


@dataclass
class FrameLabels:
    image: str
    labels: list[str]
    raw: str


def parse_labels(text: str, cap: int = 60) -> list[str]:
    """One comma-separated line, de-duplicated, order preserved.

    Greedy decoding on an off-distribution image can degenerate into a
    repeating tail, so identical and near-identical repeats are dropped and the
    list is capped — a frame with sixty distinct classes is already implausible.
    """
    line = text.strip().splitlines()[0] if text.strip() else ""
    out: list[str] = []
    seen: set[str] = set()
    for piece in line.split(","):
        label = " ".join(piece.strip().lower().split())
        if label in BANNED or label in seen:
            continue
        # a repeated phrase glued onto itself, e.g. "light fixture bulb fixture bulb"
        words = label.split()
        if len(words) > 4 or len(words) != len(set(words)):
            continue
        seen.add(label)
        out.append(label)
        if len(out) >= cap:
            break
    return out


class VLMLabeler:
    """Loads the checkpoint once and labels frames one at a time."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self._torch = torch

    def label_frame(self, image: Image.Image, previous: Sequence[str] = ()) -> FrameLabels:
        prompt = PROMPT_TEMPLATE.format(previous=", ".join(previous) if previous else "none")
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": prompt}]}
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(generated, skip_special_tokens=True)
        return FrameLabels(image="", labels=parse_labels(raw), raw=raw)

    def label_scene(
        self,
        image_paths: Iterable[str | Path],
        merge_specialisations: bool = True,
    ) -> tuple[list[FrameLabels], list[str], dict[str, str]]:
        """Label frames in order, carrying the vocabulary forward as the paper does.

        Consolidation happens *during* the pass, not after: the carried-forward
        vocabulary is what the next frame's prompt is built from, so a variant
        that slips in unchecked is repeated back to the model for every
        remaining frame. Folding it away immediately stops the cascade at the
        frame it starts on.

        Returns the per-frame labels, the vocabulary, and every merge made so
        the decision is auditable rather than silent.
        """
        vocabulary: list[str] = []
        results: list[FrameLabels] = []
        merges: dict[str, str] = {}

        for path in image_paths:
            img = Image.open(path).convert("RGB")
            res = self.label_frame(img, vocabulary)
            res.image = str(path)

            kept: list[str] = []
            for label in res.labels:
                final = consolidate(label, vocabulary) if merge_specialisations else label
                if final != label:
                    merges[label] = final
                if final not in kept:
                    kept.append(final)
                if final not in vocabulary:
                    vocabulary.append(final)
            res.labels = kept
            results.append(res)

        return results, vocabulary, merges
