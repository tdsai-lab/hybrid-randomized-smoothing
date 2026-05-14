from __future__ import annotations

import json
import os
import pandas
from typing import List, Any, Iterable, Tuple
from PIL import Image

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple, Optional

from datasets import load_dataset, load_from_disk


def _pick_path(*candidates: Optional[str]) -> str:
    cleaned = [str(Path(p)).rstrip("/") for p in candidates if p]
    for candidate in cleaned:
        if Path(candidate).exists():
            return candidate
    return cleaned[0] if cleaned else ""


def _join_candidates(roots: list[str], *suffixes: str) -> list[str]:
    out: list[str] = []
    for root in roots:
        for suffix in suffixes:
            out.append(str(Path(root) / suffix))
    return out


_DATA_ROOT_CANDIDATES = [
    os.environ.get("HYBRID_RS_DATA_ROOT"),
    "./data",
]

_HF_CACHE_CANDIDATES = [
    os.environ.get("FLICKR30K_CACHE_DIR"),
    os.environ.get("HYBRID_RS_HF_HOME"),
    os.environ.get("HF_HOME"),
    os.environ.get("TRANSFORMERS_CACHE"),
]

DEFAULT_FLICKR30K_CACHE_DIR = _pick_path(*_HF_CACHE_CANDIDATES)
DEFAULT_UNSAFEBENCH_DATASET_DIR = _pick_path(
    os.environ.get("UNSAFEBENCH_DATASET_DIR"),
    *_join_candidates(
        [root for root in _DATA_ROOT_CANDIDATES if root],
        "unsafebench_interaction_unsafe_hf",
    ),
    *_join_candidates(
        [root for root in _DATA_ROOT_CANDIDATES if root],
        "MM-SafetyBench/hf_sd",
        "MM-SafetyBench",
    ),
)
DEFAULT_MM_SAFETYBENCH_ROOT = _pick_path(
    os.environ.get("MM_SAFETYBENCH_ROOT"),
    *_join_candidates(
        [root for root in _DATA_ROOT_CANDIDATES if root],
        "MM-SafetyBench",
    ),
)
DEFAULT_HATEFUL_MEMES_DATASET_DIR = _pick_path(
    os.environ.get("HATEFUL_MEMES_DATASET_DIR"),
    *_join_candidates(
        [root for root in _DATA_ROOT_CANDIDATES if root],
        "HFdataset",
        "HarmfulMeme/HFdataset",
        "hateful_memes_hf",
    ),
)
DEFAULT_FLICKR30K_REVISION = "refs/convert/parquet"
DEFAULT_FLICKR30K_SPLIT = "test"
MM_SAFETYBENCH_IMAGE_KINDS = ("SD", "SD_TYPO", "TYPO")


# ======================================================================
# UnsafeBench interaction-mined dataset loader (text safe, image safe, joint unsafe)
# ======================================================================


# ======================================================================
# NEW: multimodal variant for Hybrid RS (text + image)
# ======================================================================
# hybrid-rs/data/mm_loader.py




@dataclass
class Flickr30kImageLoader:
    """
    Deterministic mapping from a text string -> one Flickr30k image.
    Uses a stable hash so the mapping is reproducible.

    Returns PIL.Image objects (what HF datasets store in the 'image' column).
    """
    ds: Any
    split: str = "test"

    def __post_init__(self):
        if self.split not in self.ds:
            # fall back to the first available split
            self.split = list(self.ds.keys())[0]
        self._split = self.ds[self.split]
        self._n = len(self._split)
        if self._n == 0:
            raise ValueError("Flickr30k split is empty.")

    def _index_from_text(self, text: str) -> int:
        h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, byteorder="big", signed=False) % self._n

    def __call__(self, item: dict, idx: int) -> Any:
        # You can choose item["behaviour"] or any other field; behaviour is stable.
        text = str(item.get("behaviour", "")) + f"#{idx}"
        j = self._index_from_text(text)
        return self._split[j]["image"]  # PIL image


def make_flickr30k_image_loader(
    cache_dir: str = DEFAULT_FLICKR30K_CACHE_DIR,
    revision: str = DEFAULT_FLICKR30K_REVISION,
    split: str = DEFAULT_FLICKR30K_SPLIT,
) -> Flickr30kImageLoader:
    ds = load_dataset("nlphuji/flickr30k", revision=revision, cache_dir=cache_dir)
    return Flickr30kImageLoader(ds=ds, split=split)


def get_flickr_benign(
    cache_dir: str = DEFAULT_FLICKR30K_CACHE_DIR,
    revision: str = DEFAULT_FLICKR30K_REVISION,
    split: str = DEFAULT_FLICKR30K_SPLIT,
    which_caption: int = 0,   # choose 0..4 typically
):
    """
    Yields (caption, image) from Flickr30k.
    Caption is selected from the per-image caption list.
    """
    ds = load_dataset("nlphuji/flickr30k", revision=revision, cache_dir=cache_dir)
    if split not in ds:
        split = list(ds.keys())[0]
    d = ds[split]

    for ex in d:
        img = ex["image"]
        caps = ex["caption"]
        if isinstance(caps, (list, tuple)):
            cap = caps[min(which_caption, len(caps) - 1)]
        else:
            cap = caps
        yield cap, img





def _looks_like_mm_safetybench_root(path: str | os.PathLike[str]) -> bool:
    root = Path(path).expanduser()
    return (root / "data" / "processed_questions").is_dir() and (
        root / "data" / "imgs"
    ).is_dir()


def _default_mm_safetybench_text_field(image_kind: str) -> str:
    if image_kind == "SD":
        return "Rephrased Question(SD)"
    return "Rephrased Question"


def get_mm_safetybench_mm(
    root: str = DEFAULT_MM_SAFETYBENCH_ROOT,
    *,
    image_kind: str = "SD",
    text_field: Optional[str] = None,
    max_examples: int = -1,
    target: str = "Unsafe",
) -> Iterable[Tuple[Tuple[str, Any], str]]:
    """
    Load MM-SafetyBench directly from the raw root layout.

    Returns:
        ((text, image), target)
    """
    root_path = Path(root).expanduser()
    question_dir = root_path / "data" / "processed_questions"
    image_dir = root_path / "data" / "imgs"

    if not question_dir.is_dir():
        raise FileNotFoundError(f"Missing questions directory: {question_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    if image_kind not in MM_SAFETYBENCH_IMAGE_KINDS:
        raise ValueError(
            f"Unsupported MM-SafetyBench image_kind={image_kind!r}. "
            f"Expected one of {MM_SAFETYBENCH_IMAGE_KINDS}."
        )

    text_key = text_field or _default_mm_safetybench_text_field(image_kind)
    yielded = 0

    for json_path in sorted(question_dir.glob("*.json")):
        scenario = json_path.stem
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for question_id_str, item in sorted(data.items(), key=lambda kv: int(kv[0])):
            text = (item.get(text_key, "") or "").strip()
            if not text:
                continue

            image_path = image_dir / scenario / image_kind / f"{question_id_str}.jpg"
            if not image_path.is_file():
                continue

            with Image.open(image_path) as pil_img:
                image = pil_img.convert("RGB").copy()

            yield (text, image), target
            yielded += 1

            if max_examples >= 0 and yielded >= int(max_examples):
                return


def get_unsafebench_interaction_unsafe_mm_hf(
    dataset_dir: str = DEFAULT_UNSAFEBENCH_DATASET_DIR,
    split: str = "train",
    max_examples: int = -1,
    require_mm_unsafe: bool = True,
    target: str = "Unsafe",
    target_field: Optional[str] = None,  # e.g. "category" or "safety_label"
    mm_safetybench_image_kind: str = "SD",
    mm_safetybench_text_field: Optional[str] = None,
) -> Iterable[Tuple[Tuple[str, Any], str]]:
    """
    HF on-disk DatasetDict created with Image() feature.

    Returns:
        ((text, image), target)

    target:
      - if target_field is None: constant string `target`
      - else: target = row[target_field] (falls back to constant `target` if missing/empty)
    """
    if _looks_like_mm_safetybench_root(dataset_dir):
        yield from get_mm_safetybench_mm(
            root=dataset_dir,
            image_kind=mm_safetybench_image_kind,
            text_field=mm_safetybench_text_field,
            max_examples=max_examples,
            target=target,
        )
        return

    ds = load_from_disk(dataset_dir)

    if split not in ds:
        split = list(ds.keys())[0]
    d = ds[split]

    n_total = len(d) if max_examples < 0 else min(len(d), max_examples)

    for idx in range(n_total):
        row = d[idx]

        if require_mm_unsafe and (not bool(row.get("mm_unsafe", False))):
            continue

        text = (row.get("text", "") or "").strip()
        if not text:
            continue

        image = row.get("image", None)  # PIL.Image.Image (decoded by HF)
        if image is None:
            continue

        if target_field is None:
            y = target
        else:
            y0 = row.get(target_field, None)
            y = target if (y0 is None or str(y0).strip() == "") else str(y0)

        yield (text, image), y


def get_hateful_memes_mm(
    dataset_dir: str = DEFAULT_HATEFUL_MEMES_DATASET_DIR,
    split: Optional[str] = None,   # dataset is a single Dataset, not a DatasetDict
    max_examples: int = -1,
    target: str = "Unsafe",
    target_field: Optional[str] = "label",  # or None
    require_label: Optional[int] = 1,        # keep only label==1 if not None
) -> Iterable[Tuple[Tuple[str, Any], str]]:
    """
    Loader for the HF dataset created from Final_Unsafe_samples.json.

    Dataset schema (as created earlier):
      - id: int
      - img: Image()        -> PIL.Image.Image
      - label: ClassLabel(["safe","unsafe"]) or int
      - text: str
      - SafeScore: float
      - UnsafeScore: float

    Returns:
        ((text, image), target)

    target:
      - if target_field is None: constant string `target`
      - else: target = row[target_field] (stringified)
    """
    ds = load_from_disk(dataset_dir)

    # handle both Dataset and DatasetDict defensively
    if hasattr(ds, "keys") and split is not None and split in ds:
        d = ds[split]
    else:
        d = ds

    n_total = len(d) if max_examples < 0 else min(len(d), max_examples)

    for idx in range(n_total):
        row = d[idx]

        # optional filtering on label
        if require_label is not None:
            lab = row.get("label", None)
            try:
                if int(lab) != int(require_label):
                    continue
            except Exception:
                continue

        text = (row.get("text", "") or "").strip()
        if not text:
            continue

        image = row.get("img", None)  # HF Image() -> PIL.Image.Image
        if image is None:
            continue

        if target_field is None:
            y = target
        else:
            y0 = row.get(target_field, None)
            y = target if y0 is None else str(y0)

        yield (text, image), y


def add_multimodal_dataset_args(
    parser,
    *,
    default_dataset: str = "hateful_memes_mm",
    include_hateful_memes: bool = True,
):
    dataset_choices = ["unsafebench_interaction_unsafe_mm"]
    if include_hateful_memes:
        dataset_choices.append("hateful_memes_mm")

    parser.add_argument(
        "--dataset",
        type=str,
        default=default_dataset,
        choices=dataset_choices,
    )
    parser.add_argument(
        "--unsafebench_dataset_dir",
        type=str,
        default=DEFAULT_UNSAFEBENCH_DATASET_DIR,
        help="Path to the on-disk UnsafeBench interaction dataset or a raw MM-SafetyBench root.",
    )
    parser.add_argument(
        "--mm_safetybench_image_kind",
        type=str,
        default="SD",
        choices=list(MM_SAFETYBENCH_IMAGE_KINDS),
        help="Image subfolder to use when unsafebench_dataset_dir points to a raw MM-SafetyBench root.",
    )
    parser.add_argument(
        "--mm_safetybench_text_field",
        type=str,
        default=None,
        help="Question field to use when unsafebench_dataset_dir points to a raw MM-SafetyBench root.",
    )
    if include_hateful_memes:
        parser.add_argument(
            "--hateful_memes_dataset_dir",
            type=str,
            default=DEFAULT_HATEFUL_MEMES_DATASET_DIR,
            help="Path to the on-disk Hateful Memes HF dataset.",
        )
        parser.add_argument(
            "--hateful_memes_require_label",
            type=int,
            default=1,
            help=(
                "Label value required for hateful_memes_mm. "
                "Set to a negative value to disable label filtering."
            ),
        )
    parser.add_argument(
        "--flickr_cache_dir",
        type=str,
        default=DEFAULT_FLICKR30K_CACHE_DIR,
        help="HF cache directory used for Flickr30k.",
    )
    parser.add_argument(
        "--flickr_revision",
        type=str,
        default=DEFAULT_FLICKR30K_REVISION,
        help="Dataset revision for nlphuji/flickr30k.",
    )
    parser.add_argument(
        "--flickr_split",
        type=str,
        default=DEFAULT_FLICKR30K_SPLIT,
        help="Dataset split for nlphuji/flickr30k.",
    )


def get_multimodal_dataset_loader_from_args(args, *, include_hateful_memes: bool = True):
    if args.dataset == "unsafebench_interaction_unsafe_mm":
        return get_unsafebench_interaction_unsafe_mm_hf(
            dataset_dir=args.unsafebench_dataset_dir,
            split="train",
            max_examples=args.max_examples,
            require_mm_unsafe=True,
            mm_safetybench_image_kind=args.mm_safetybench_image_kind,
            mm_safetybench_text_field=args.mm_safetybench_text_field,
        )

    if args.dataset == "hateful_memes_mm":
        if not include_hateful_memes:
            raise ValueError("hateful_memes_mm is not enabled for this entrypoint.")
        require_label = int(args.hateful_memes_require_label)
        return get_hateful_memes_mm(
            dataset_dir=args.hateful_memes_dataset_dir,
            split="train",
            max_examples=args.max_examples,
            require_label=None if require_label < 0 else require_label,
        )

    raise ValueError(f"Unknown dataset: {args.dataset!r}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    
    loader = get_unsafebench_interaction_unsafe_mm_hf(
        dataset_dir=DEFAULT_UNSAFEBENCH_DATASET_DIR,
        split="train",
        target="Unsafe",
        # or: target_field="category",
    )
    print("length loader:", len(list(loader)))

    (text, image), target = next(iter(loader))
    print(type(image), target, text[:80])   
