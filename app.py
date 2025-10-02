import os
import faiss
import torch
import numpy as np
import requests
from datetime import datetime
from torchvision import models, transforms
from flask import Flask, request, render_template, send_from_directory, jsonify
from PIL import Image
import shutil
import tempfile

# ============================
# Config
# ============================
UNSPLASH_ACCESS_KEY = "g2YPsiwAcq4xa3-zLthhXlA-3NF4u7KeCT_IqXwNJqY"  # Thay bằng key thật
TEMP_FOLDER = "temp_images"
LOCAL_IMAGE_FOLDER = "images"
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(LOCAL_IMAGE_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")

# ============================
# 1. Model ResNet18 (Cải thiện error handling)
# ============================
model = models.resnet18(weights="IMAGENET1K_V1")
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_feature(img_path):
    try:
        # Test open PIL trước
        with Image.open(img_path) as img_pil:
            img = img_pil.convert("RGB")  # Force RGB để tránh format issue
            print(f"   ✅ PIL open OK: {img_path} (size: {img.size})")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_tensor).squeeze().cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        print(f"   ✅ Extract feature OK: {img_path}")
        return feat
    except Exception as e:
        print(f"   ❌ Extract fail {img_path}: {e}")
        return None

# ============================
# 2. Load local DB (Thêm debug)
# ============================
def load_local_db():
    image_paths = [os.path.join(LOCAL_IMAGE_FOLDER, f) for f in os.listdir(LOCAL_IMAGE_FOLDER)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    features = []
    metadata_db = {}
    print("🔍 Load local DB...")
    print(f"   Local paths: {[os.path.basename(p) for p in image_paths]}")
    
    for idx, path in enumerate(image_paths):
        vec = extract_feature(path)
        if vec is not None:
            features.append(vec)
            metadata_db[idx] = {
                "filename": os.path.basename(path),
                "year": 2023 if idx % 2 == 0 else 2024,
                "author": "Local User",
                "description": "Ảnh local",
                "tags": "local",
                "location": "N/A",
                "created_at": "2024-01-01",
                "download_url": f"/images/{os.path.basename(path)}"
            }
    
    if features:
        features = np.array(features).astype("float32")
        d = features.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(features)
        print(f"✅ Local DB: {len(features)} ảnh loaded.")
        return index, metadata_db
    else:
        print("⚠ Local DB rỗng (thêm ảnh vào images/).")
        d = 512
        index = faiss.IndexFlatL2(d)
        return index, {}

# ============================
# 3. Unsplash API (Cải thiện: Debug + partial success)
# ============================
def search_unsplash(query, per_page=5):
    if not UNSPLASH_ACCESS_KEY or UNSPLASH_ACCESS_KEY == "YOUR_UNSPLASH_ACCESS_KEY_HERE":
        print("⚠ No API key, skip Unsplash.")
        return None, {}, []
    
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}
    params = {"query": query, "per_page": per_page, "orientation": "squarish"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        print(f"📡 Unsplash API: {len(results)} photos found.")
        
        features = []
        metadata_db = {}
        downloaded_paths = []
        
        print(f"⬇️ Downloading & extracting {len(results)} images...")
        for idx, photo in enumerate(results):
            try:
                created_at = photo.get("created_at", "")
                year = int(datetime.fromisoformat(created_at).year) if created_at else 2023
                user = photo.get("user", {})
                author = user.get("name", "Unknown")
                description = photo.get("description", "") or photo.get("alt_description", "")
                tags = [keyword["slug"] for keyword in photo.get("tags", [])[:3]]
                location = photo.get("location", {}).get("title", "N/A") if photo.get("location") else "N/A"
                download_url = photo["urls"]["small"]
                
                # Download
                print(f"   Downloading {idx+1}: {photo['id']}")
                img_response = requests.get(download_url, stream=True)
                img_response.raise_for_status()
                filename = f"{photo['id']}.jpg"
                img_path = os.path.join(TEMP_FOLDER, filename)
                with open(img_path, "wb") as f:
                    for chunk in img_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Check file
                if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                    downloaded_paths.append(img_path)
                    print(f"   📁 Saved: {filename} (size: {os.path.getsize(img_path)} bytes)")
                    
                    # Extract (có thể fail)
                    vec = extract_feature(img_path)
                    if vec is not None:
                        features.append(vec)
                        metadata_db[idx] = {
                            "filename": filename,
                            "year": year,
                            "author": author,
                            "description": description,
                            "tags": ", ".join(tags),
                            "location": location,
                            "created_at": created_at,
                            "download_url": photo["urls"]["full"]
                        }
                        print(f"   ✅ Processed: {filename}")
                    else:
                        print(f"   ⚠ Skipped extract: {filename} (keep metadata for keyword search)")
                        # VẪN THÊM METADATA ĐỂ DÙNG CHO KEYWORD SEARCH (FIX BUG)
                        metadata_db[idx] = {
                            "filename": filename,
                            "year": year,
                            "author": author,
                            "description": description,
                            "tags": ", ".join(tags),
                            "location": location,
                            "created_at": created_at,
                            "download_url": photo["urls"]["full"]
                        }
                else:
                    print(f"   ❌ Download fail: {filename}")
                    if os.path.exists(img_path):
                        os.remove(img_path)
            except Exception as e:
                print(f"   ❌ Error {idx}: {e}")
        
        # Build index nếu có features (có thể partial)
        if features:
            features = np.array(features).astype("float32")
            d = features.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(features)
            print(f"✅ Unsplash index: {len(features)}/{len(results)} processed.")
            return index, metadata_db, downloaded_paths
        else:
            print("⚠ No features extracted, but metadata available for keyword search.")
            # Return dummy index nếu chỉ keyword search (không upload)
            d = 512
            index = faiss.IndexFlatL2(d)
            return index, metadata_db, downloaded_paths  # FIX: Return metadata dù no features
    except Exception as e:
        print(f"❌ Unsplash API error: {e}")
        return None, {}, []

# ============================
# 4. Flask App
# ============================
app = Flask(__name__)

# Global
current_index = None
current_metadata = {}
current_paths = []
use_unsplash = False

@app.route("/", methods=["GET", "POST"])
def home():
    global current_index, current_metadata, current_paths, use_unsplash
    
    error = None
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            error = "⚠ Vui lòng nhập từ khóa!"
            return render_template("index.html", error=error)
        
        # Clear temp
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        
        # Thử Unsplash
        print("🚀 Starting search for:", keyword)
        current_index, current_metadata, current_paths = search_unsplash(keyword)
        
        # FIX LOGIC: Nếu có metadata từ Unsplash (dù index partial), dùng Unsplash mode
        if current_metadata:  # Priority: Nếu có data từ Unsplash
            use_unsplash = True
            print(f"✅ Use Unsplash mode: {len(current_metadata)} items")
        elif current_index is None:  # Fallback chỉ nếu hoàn toàn fail
            current_index, current_metadata = load_local_db()
            use_unsplash = False
            print("🔄 Fallback to local.")
        else:
            use_unsplash = True  # Nếu index OK
        
        print(f"📊 Final: Use Unsplash={use_unsplash}, Metadata keys={[k for k in current_metadata.keys()][:3]}...")  # Debug filenames
        
        if not current_metadata:
            error = "⚠ Không có dữ liệu (thêm ảnh vào images/ hoặc kiểm tra API key)."
            return render_template("index.html", error=error)
        
        # Upload query (giữ nguyên + debug từ trước)
        file = request.files.get("file")
        query_filename = None
        if file and file.filename != "":
            if not file.content_type or not file.content_type.startswith('image/'):
                error = "⚠ File không phải ảnh!"
                return render_template("index.html", error=error)
            
            timestamp = int(datetime.now().timestamp())
            query_filename = f"query_{timestamp}.jpg"
            query_path = os.path.join(TEMP_FOLDER, query_filename)
            try:
                file.save(query_path)
                if os.path.exists(query_path) and os.path.getsize(query_path) > 100:
                    print(f"✅ Query saved: {query_path} (size: {os.path.getsize(query_path)})")
                else:
                    raise Exception("File rỗng")
            except Exception as e:
                print(f"❌ Query save error: {e}")
                error = "⚠ Lỗi lưu ảnh query!"
                return render_template("index.html", error=error)
            
            # Search similar (chỉ nếu index có data)
            if current_index.ntotal > 0:
                qvec = extract_feature(query_path)
                if qvec is not None:
                    qvec = qvec.astype("float32").reshape(1, -1)
                    k = min(5, current_index.ntotal)
                    D, I = current_index.search(qvec, k)
                    
                    results = []
                    for dist, idx in zip(D[0], I[0]):
                        if idx < len(current_metadata):
                            meta = current_metadata[idx]
                            similarity = round((1 / (1 + dist)) * 100, 2)
                            meta_score = round(100 if keyword.lower() in meta.get("tags", "").lower() else 50, 2)
                            results.append({**meta, "similarity": similarity, "meta_score": meta_score})
                    results = sorted(results, key=lambda x: x["similarity"] + x["meta_score"], reverse=True)[:5]
                else:
                    results = []  # No search if query fail
            else:
                results = list(current_metadata.values())[:5]  # Fallback top
                for r in results:
                    r["similarity"] = 0
                    r["meta_score"] = 100
            return render_template("results.html", query=query_filename, results=results, keyword=keyword, use_unsplash=use_unsplash)
        
        else:
            # No upload: Top results
            results = list(current_metadata.values())[:5]
            for r in results:
                r["similarity"] = 0
                r["meta_score"] = 100
            return render_template("results.html", query=None, results=results, keyword=keyword, use_unsplash=use_unsplash)
    
    return render_template("index.html", error=error)

@app.route("/images/<filename>")
def images_static(filename):
    full_path = os.path.join(LOCAL_IMAGE_FOLDER, filename)
    print(f"🔍 Local serve: {full_path} (exist: {os.path.exists(full_path)})")
    if os.path.exists(full_path):
        return send_from_directory(LOCAL_IMAGE_FOLDER, filename)
    return "Not found", 404

@app.route("/temp_images/<filename>")
def temp_images_static(filename):
    full_path = os.path.join(TEMP_FOLDER, filename)
    print(f"🔍 Temp serve: {full_path} (exist: {os.path.exists(full_path)})")
    if os.path.exists(full_path):
        return send_from_directory(TEMP_FOLDER, filename)
    return "Not found", 404

@app.route("/cleanup")
def cleanup():
    global current_paths
    for path in current_paths:
        if os.path.exists(path):
            os.remove(path)
    current_paths = []
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)
    return "Cleaned!"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)