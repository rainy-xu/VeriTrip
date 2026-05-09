import torch
import faiss
import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from typing import List, Dict, Any
# from .. dinov2.hub.backbones import dinov2_vits14

class ImageIndexingManager:
    def __init__(self, root_dir: str, index_path: str = "data.bin", embedding_path: str = "all_embeddings.json"):
        self.root_dir = root_dir
        self.index_path = index_path
        self.embedding_path = embedding_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.environ.get("DINOv2_CHECKPOINT_PATH", "")

        dinov2_source = os.environ.get("DINOv2_SOURCE_DIR", "")
        self.model = torch.hub.load(dinov2_source, "dinov2_vits14", source='local',
                            pretrained=False)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        
        self.transform = T.Compose([
            T.ToTensor(), 
            T.Resize(244), 
            T.CenterCrop(224), 
            T.Normalize([0.5], [0.5])
        ])
        
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.files.sort() 
        

        if os.path.exists(index_path):
            print(f"Loading existing index from {index_path}")
            self.index = faiss.read_index(index_path)
            with open(embedding_path, "r") as f:
                self.all_embeddings = json.load(f)
        else:
            print("Index not found. Creating new index (this may take time)...")
            self.index, self.all_embeddings = self._create_index()

    def _load_image(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    def _create_index(self):
        index = faiss.IndexFlatL2(384) # dinov2_vits14 维度是 384
        all_embeddings = {}
        
        with torch.no_grad():
            for file in tqdm(self.files):
                emb = self.model(self._load_image(file))
                embedding = emb[0].cpu().numpy().reshape(1, -1)
                all_embeddings[file] = embedding.tolist()
                index.add(embedding)
        
        faiss.write_index(index, self.index_path)
        with open(self.embedding_path, "w") as f:
            json.dump(all_embeddings, f)
        return index, all_embeddings

    def search_similar_by_path(self, query_img_path: str, k: int = 3) -> List[Dict[str, Any]]:
        if not os.path.exists(query_img_path):
            return [{"error": f"Image path {query_img_path} not found"}]
        
        with torch.no_grad():
            emb = self.model(self._load_image(query_img_path))
            query_vector = emb[0].cpu().numpy().reshape(1, -1)
            
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "rank": i + 1,
                "file_path": self.files[idx],
                "distance": float(distances[0][i])
            })
        return results




if __name__ == "__main__":
    image_retriver = ImageIndexingManager(
        root_dir=os.environ.get("IMAGE_ROOT_DIR", ""),
        index_path=os.environ.get("IMAGE_INDEX_PATH", ""),
        embedding_path=os.environ.get("IMAGE_EMBEDDING_PATH", ""))
    # results = image_retriver.search_similar_by_id('31',3)

    # print("similar indices", indices_result)
