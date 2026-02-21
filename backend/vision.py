"""
vision.py — image analysis using Groq's vision-capable LLaMA 4 Scout
Returns description, tags, mood, and dominant colors per image
"""
import os
import base64
import json
from groq import Groq
from PIL import Image
import io

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def image_to_base64(image_bytes: bytes) -> tuple[str, str]:
    """Convert image bytes to base64, normalizing format to JPEG."""
    img = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB (handles RGBA, palette images, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return b64, "image/jpeg"

def analyze_image(image_bytes: bytes, filename: str) -> dict:
    """
    Send image to Groq vision model and get back structured analysis.
    Returns: description, tags, mood, colors, scene_type
    """
    b64, mime = image_to_base64(image_bytes)

    prompt = """Analyze this image and respond with ONLY a valid JSON object, no markdown, no explanation.

{
  "description": "2-3 sentence natural description of what you see",
  "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
  "mood": "one word mood/atmosphere (e.g. peaceful, energetic, melancholic, vibrant)",
  "scene_type": "one of: portrait, landscape, urban, nature, food, animal, object, abstract, architecture, other",
  "dominant_colors": ["color1", "color2", "color3"],
  "time_of_day": "day/night/sunset/indoor/unknown"
}

Tags should be specific and searchable (objects, activities, settings, styles).
Respond with ONLY the JSON, nothing else."""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=400,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        result["filename"] = filename
        result["status"] = "ok"
        return result

    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return {
            "filename": filename,
            "description": raw if 'raw' in dir() else "Analysis failed",
            "tags": [],
            "mood": "unknown",
            "scene_type": "other",
            "dominant_colors": [],
            "time_of_day": "unknown",
            "status": "partial"
        }
    except Exception as e:
        return {
            "filename": filename,
            "description": f"Error: {str(e)}",
            "tags": [],
            "mood": "unknown",
            "scene_type": "other",
            "dominant_colors": [],
            "time_of_day": "unknown",
            "status": "error"
        }
