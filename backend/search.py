"""
search.py — natural language search over image analyses
Simple but effective: searches descriptions, tags, mood, scene_type
No embeddings needed — the descriptions are rich enough for keyword + fuzzy matching
"""

def search_images(query: str, analyses: dict) -> list[str]:
    """
    Search images by natural language query.
    Returns list of image_ids sorted by relevance score.
    
    analyses: { image_id: { description, tags, mood, scene_type, ... } }
    """
    query_words = set(query.lower().split())
    scored = []

    for image_id, data in analyses.items():
        score = 0.0

        # Search in description (highest weight)
        desc_words = set(data.get("description", "").lower().split())
        matches = query_words & desc_words
        score += len(matches) * 3.0

        # Search in tags
        tags = [t.lower() for t in data.get("tags", [])]
        for tag in tags:
            for qw in query_words:
                if qw in tag or tag in qw:
                    score += 2.0

        # Mood match
        mood = data.get("mood", "").lower()
        if any(qw in mood for qw in query_words):
            score += 2.0

        # Scene type match
        scene = data.get("scene_type", "").lower()
        if any(qw in scene for qw in query_words):
            score += 2.0

        # Time of day match
        tod = data.get("time_of_day", "").lower()
        if any(qw in tod for qw in query_words):
            score += 1.5

        # Color match
        colors = [c.lower() for c in data.get("dominant_colors", [])]
        for color in colors:
            if any(qw in color for qw in query_words):
                score += 1.0

        if score > 0:
            scored.append((image_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [img_id for img_id, _ in scored]
