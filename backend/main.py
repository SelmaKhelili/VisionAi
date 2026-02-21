"""
main.py — FastAPI backend for VisionAI
Endpoints:
  POST /upload          — upload images, analyze with vision LLM
  GET  /images          — list all images with analyses
  GET  /images/{id}     — get single image data
  GET  /images/{id}/file — serve the actual image file
  POST /search          — natural language search
  GET  /clusters        — get clustered image groups
  DELETE /images/{id}   — remove an image
  GET  /                — serve frontend
"""
import os
import uuid
import base64
import pathlib
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio

load_dotenv()

from vision import analyze_image, image_to_base64
from clustering import extract_features, cluster_images, get_cluster_label
from search import search_images

BASE_DIR = pathlib.Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# In-memory store
# { image_id: { "filename", "bytes", "b64_thumb", "analysis", "features", "cluster" } }
image_store: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── Upload & Analyze ──────────────────────────────────────────
@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    results = []
    new_ids = []

    for file in files:
        ext = pathlib.Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({"filename": file.filename, "status": "error", "message": "Unsupported format"})
            continue

        contents = await file.read()
        image_id = str(uuid.uuid4())

        # Generate thumbnail base64 for frontend display
        try:
            b64, mime = image_to_base64(contents)
            b64_thumb = f"data:{mime};base64,{b64}"
        except Exception as e:
            results.append({"filename": file.filename, "status": "error", "message": str(e)})
            continue

        # Analyze with vision LLM
        analysis = await asyncio.to_thread(analyze_image, contents, file.filename)

        # Extract clustering features
        features = await asyncio.to_thread(extract_features, contents, analysis)

        image_store[image_id] = {
            "filename": file.filename,
            "bytes": contents,
            "b64_thumb": b64_thumb,
            "analysis": analysis,
            "features": features,
            "cluster": 0,
        }
        new_ids.append(image_id)
        results.append({"filename": file.filename, "image_id": image_id, "status": "ok"})

    # Re-cluster all images whenever new ones are added
    if new_ids and len(image_store) >= 2:
        cluster_data = [
            {"image_id": iid, "features": img["features"]}
            for iid, img in image_store.items()
        ]
        clustered = cluster_images(cluster_data)
        for item in clustered:
            image_store[item["image_id"]]["cluster"] = item["cluster"]

    return {"results": results}


# ── List images ───────────────────────────────────────────────
@app.get("/images")
async def list_images():
    return {
        "images": [
            {
                "image_id": iid,
                "filename": img["filename"],
                "b64_thumb": img["b64_thumb"],
                "analysis": img["analysis"],
                "cluster": img["cluster"],
            }
            for iid, img in image_store.items()
        ]
    }


# ── Single image ──────────────────────────────────────────────
@app.get("/images/{image_id}")
async def get_image(image_id: str):
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Image not found")
    img = image_store[image_id]
    return {
        "image_id": image_id,
        "filename": img["filename"],
        "b64_thumb": img["b64_thumb"],
        "analysis": img["analysis"],
        "cluster": img["cluster"],
    }


# ── Delete image ──────────────────────────────────────────────
@app.delete("/images/{image_id}")
async def delete_image(image_id: str):
    if image_id not in image_store:
        raise HTTPException(status_code=404, detail="Not found")
    del image_store[image_id]
    # Re-cluster remaining
    if len(image_store) >= 2:
        cluster_data = [{"image_id": iid, "features": img["features"]} for iid, img in image_store.items()]
        clustered = cluster_images(cluster_data)
        for item in clustered:
            image_store[item["image_id"]]["cluster"] = item["cluster"]
    return {"status": "ok"}


# ── Search ────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str

@app.post("/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        return {"results": list(image_store.keys())}
    analyses = {iid: img["analysis"] for iid, img in image_store.items()}
    ranked_ids = search_images(req.query, analyses)
    results = []
    for iid in ranked_ids:
        img = image_store[iid]
        results.append({
            "image_id": iid,
            "filename": img["filename"],
            "b64_thumb": img["b64_thumb"],
            "analysis": img["analysis"],
            "cluster": img["cluster"],
        })
    return {"results": results}


# ── Clusters ──────────────────────────────────────────────────
@app.get("/clusters")
async def get_clusters():
    if not image_store:
        return {"clusters": []}

    # Group by cluster id
    groups: dict[int, list] = {}
    for iid, img in image_store.items():
        c = img["cluster"]
        if c not in groups:
            groups[c] = []
        groups[c].append({
            "image_id": iid,
            "filename": img["filename"],
            "b64_thumb": img["b64_thumb"],
            "analysis": img["analysis"],
        })

    clusters = []
    for cluster_id, images in sorted(groups.items()):
        analyses = [img["analysis"] for img in images]
        label = get_cluster_label(analyses)
        clusters.append({
            "cluster_id": cluster_id,
            "label": label,
            "count": len(images),
            "images": images,
        })

    return {"clusters": clusters}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
