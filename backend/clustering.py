"""
clustering.py — hybrid semantic clustering using CLIP + text embeddings
Blends two signals 50/50:
  - CLIP image embedding (512-dim): how the image looks / photographic style
  - Sentence-transformers text embedding (384-dim): what the image means / content

This prevents CLIP from over-clustering on composition style alone.
Two burger photos with different angles AND two "breakfast food" photos
will now correctly end up in different clusters.
"""
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

# ── Load CLIP ─────────────────────────────────────────────────
print("Loading CLIP model...")
try:
    from transformers import CLIPProcessor, CLIPModel
    import torch
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    CLIP_AVAILABLE = True
    print("CLIP ready ✓")
except Exception as e:
    print(f"CLIP unavailable: {e}")
    CLIP_AVAILABLE = False

# ── Load sentence-transformers for text ───────────────────────
print("Loading text embedding model...")
try:
    from sentence_transformers import SentenceTransformer
    text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    TEXT_AVAILABLE = True
    print("Text embedder ready ✓")
except Exception as e:
    print(f"Text embedder unavailable: {e}")
    TEXT_AVAILABLE = False


def _clip_embedding(image_bytes: bytes) -> np.ndarray | None:
    """512-dim CLIP image embedding, L2 normalized."""
    if not CLIP_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        emb = features[0].numpy().flatten().astype(np.float64)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
    except Exception as e:
        print(f"CLIP error: {e}")
        return None


def _text_embedding(analysis: dict) -> np.ndarray | None:
    """384-dim text embedding of description + tags, L2 normalized."""
    if not TEXT_AVAILABLE:
        return None
    try:
        # Build a rich text representation from the analysis
        parts = []
        if analysis.get("description"):
            parts.append(analysis["description"])
        if analysis.get("tags"):
            parts.append(", ".join(analysis["tags"]))
        if analysis.get("scene_type"):
            parts.append(analysis["scene_type"])
        if analysis.get("mood"):
            parts.append(analysis["mood"])
        text = ". ".join(parts) if parts else "unknown"
        emb = text_embedder.encode([text], show_progress_bar=False)[0].astype(np.float64)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
    except Exception as e:
        print(f"Text embed error: {e}")
        return None


def _color_histogram(image_bytes: bytes) -> np.ndarray:
    """Fallback: 48-dim color histogram."""
    features = []
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
        for channel in img.split():
            hist = channel.histogram()
            binned = [sum(hist[i*16:(i+1)*16]) for i in range(16)]
            total = sum(binned) or 1
            features.extend([x / total for x in binned])
    except:
        features.extend([0.0] * 48)
    return np.array(features, dtype=np.float64)


def extract_features(image_bytes: bytes, analysis: dict) -> np.ndarray:
    """
    Build a hybrid feature vector blending CLIP + text embeddings.
    
    Strategy:
    - If both available: concatenate [CLIP(512) | text(384)] with equal weight
      (both are L2 normalized so they contribute equally)
    - If only CLIP: use CLIP alone
    - If only text: use text alone  
    - If neither: fall back to color histogram
    
    Concatenation is better than averaging here because it preserves
    both signals independently for KMeans to use.
    """
    clip_emb = _clip_embedding(image_bytes)
    text_emb = _text_embedding(analysis)

    if clip_emb is not None and text_emb is not None:
        # Concatenate — both normalized so scale is equal
        return np.concatenate([clip_emb, text_emb])
    elif clip_emb is not None:
        return clip_emb
    elif text_emb is not None:
        return text_emb
    else:
        return _color_histogram(image_bytes)


def cluster_images(image_data: list[dict]) -> list[dict]:
    """
    K-means clustering on hybrid feature vectors.
    k heuristic: sqrt(n/2) + 1, capped 2-8.
    """
    n = len(image_data)
    if n < 2:
        for item in image_data:
            item["cluster"] = 0
        return image_data

    k = max(2, min(8, int((n / 2) ** 0.5) + 1))

    matrix = np.array([d["features"] for d in image_data], dtype=np.float64)

    # Guard against NaN/inf
    if not np.all(np.isfinite(matrix)):
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    k = min(k, len(matrix))
    if k < 2:
        for item in image_data:
            item["cluster"] = 0
        return image_data

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrix)

    for item, label in zip(image_data, labels):
        item["cluster"] = int(label)

    return image_data


def get_cluster_label(analyses: list[dict]) -> str:
    """Top 3 tags from a cluster as a human-readable label."""
    from collections import Counter
    all_tags = []
    for a in analyses:
        all_tags.extend(a.get("tags", []))
    common = Counter(all_tags).most_common(3)
    return " · ".join(t for t, _ in common) if common else "Mixed"