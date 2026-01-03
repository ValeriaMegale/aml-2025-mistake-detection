"""
TaskGraphMatcher: Modello per Task Verification tramite Graph Matching

Questo modulo implementa l'architettura per:
1. Encoding testuale delle descrizioni del Task Graph (usando sentence-transformers)
2. Matching tra step visivi e nodi del task graph (Hungarian algorithm)
3. Classification finale per verificare se la procedura è stata eseguita correttamente

Extension "From Mistake Detection to Task Verification" - Substep 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class TaskGraphMatcher(nn.Module):
    """
    Modello per Task Verification tramite matching tra step visivi e task graph.
    
    Architettura:
    - Text Encoder: Encoda le descrizioni testuali dei nodi del task graph
    - Visual Projection: Proietta gli embedding visivi nello spazio condiviso
    - Matching Layer: Calcola similarità e applica Hungarian matching
    - Classification Head: Predice se la procedura è corretta
    
    Supporta due configurazioni:
    1. Perception/CLIP (default): visual_dim=768, text_dim=512
    2. Omnivore/sentence-transformers: visual_dim=1024, text_dim=384
    """
    
    def __init__(
        self, 
        visual_dim=768,           # Dimensione embedding Perception (768) o Omnivore (1024)
        text_dim=512,             # Dimensione CLIP text (512) o sentence-transformers (384)
        hidden_dim=256,           # Dimensione spazio condiviso
        n_heads=4,
        n_layers=2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Proiezione Visual -> Hidden
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Proiezione Text -> Hidden
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Transformer per contestualizzare gli step visivi
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.visual_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Positional Encoding per step visivi
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, hidden_dim))
        
        # Cross-attention: Visual attende ai nodi del task graph
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Layer per fondere matching scores con features
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),  # +1 per matching score
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification Head (video-level)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        visual_emb,           # [B, N_steps, visual_dim] - step embeddings
        text_emb,             # [B, N_nodes, text_dim] - task graph node embeddings
        visual_mask=None,     # [B, N_steps] - True dove padding
        text_mask=None        # [B, N_nodes] - True dove padding
    ):
        """
        Forward pass con Hungarian matching.
        
        Returns:
            prob: [B, 1] - probabilità che il video contenga errori
            matching_cost: [B] - costo del matching ottimale (per analisi)
        """
        B = visual_emb.size(0)
        N_steps = visual_emb.size(1)
        N_nodes = text_emb.size(1)
        
        # 1. Proiezione nello spazio condiviso
        vis_proj = self.visual_proj(visual_emb)  # [B, N_steps, hidden]
        txt_proj = self.text_proj(text_emb)      # [B, N_nodes, hidden]
        
        # 2. Positional encoding per visual
        if N_steps <= self.pos_embedding.size(1):
            vis_proj = vis_proj + self.pos_embedding[:, :N_steps, :]
        
        # 3. Self-attention sui visual steps
        vis_ctx = self.visual_transformer(vis_proj, src_key_padding_mask=visual_mask)
        
        # 4. Cross-attention: Visual queries, Text keys/values
        vis_attended, attn_weights = self.cross_attention(
            query=vis_ctx,
            key=txt_proj,
            value=txt_proj,
            key_padding_mask=text_mask
        )  # vis_attended: [B, N_steps, hidden]
        
        # 5. Calcolo matrice di similarità per Hungarian matching
        # Normalizza per cosine similarity
        vis_norm = F.normalize(vis_ctx, p=2, dim=-1)
        txt_norm = F.normalize(txt_proj, p=2, dim=-1)
        
        similarity = torch.bmm(vis_norm, txt_norm.transpose(1, 2))  # [B, N_steps, N_nodes]
        
        # 6. Hungarian Matching (offline, per ogni sample nel batch)
        matching_costs = []
        matched_text_features = []
        match_scores = []
        
        for b in range(B):
            # Cost matrix: negativo della similarità (Hungarian minimizza)
            cost_matrix = -similarity[b].detach().cpu().numpy()
            
            # Maschera per padding
            if visual_mask is not None:
                valid_steps = (~visual_mask[b]).sum().item()
            else:
                valid_steps = N_steps
                
            if text_mask is not None:
                valid_nodes = (~text_mask[b]).sum().item()
            else:
                valid_nodes = N_nodes
            
            # Hungarian matching solo su parti valide
            row_ind, col_ind = linear_sum_assignment(
                cost_matrix[:int(valid_steps), :int(valid_nodes)]
            )
            
            # Costo totale del matching
            total_cost = cost_matrix[row_ind, col_ind].sum()
            matching_costs.append(-total_cost / max(len(row_ind), 1))  # Negativo per avere similarità
            
            # Raccogli i text features matchati
            matched_txt = torch.zeros(N_steps, self.hidden_dim, device=visual_emb.device)
            scores = torch.zeros(N_steps, 1, device=visual_emb.device)
            
            for i, (r, c) in enumerate(zip(row_ind, col_ind)):
                matched_txt[r] = txt_proj[b, c]
                scores[r, 0] = similarity[b, r, c]
            
            matched_text_features.append(matched_txt)
            match_scores.append(scores)
        
        matching_costs = torch.tensor(matching_costs, device=visual_emb.device)
        matched_text_features = torch.stack(matched_text_features)  # [B, N_steps, hidden]
        match_scores = torch.stack(match_scores)  # [B, N_steps, 1]
        
        # 7. Fusione features
        fused = torch.cat([vis_ctx, matched_text_features, match_scores], dim=-1)
        fused = self.fusion_layer(fused)  # [B, N_steps, hidden]
        
        # 8. Aggregazione (masked mean pooling)
        if visual_mask is not None:
            valid_mask = (~visual_mask).unsqueeze(-1).float()
            fused_sum = (fused * valid_mask).sum(dim=1)
            valid_count = valid_mask.sum(dim=1).clamp(min=1.0)
            pooled = fused_sum / valid_count
        else:
            pooled = fused.mean(dim=1)
        
        # 9. Classification
        prob = self.classifier(pooled)
        
        return prob, matching_costs


class TextEncoder:
    """
    Text encoder per le descrizioni del task graph.
    
    Supporta due modalità:
    1. CLIP (default): Usa OpenAI CLIP per embeddings allineati video-text (768-dim)
       - Compatibile con feature "perception" che sono basate su CLIP/ViT
    2. sentence-transformers: Usa modelli sentence-transformers (384-dim default)
    
    Per l'extension, la specifica richiede EgoVLP/PE con spazi allineati,
    quindi CLIP è la scelta consigliata quando si usano feature perception.
    """
    
    def __init__(self, model_name='clip', device=None):
        """
        Args:
            model_name: 'clip' per CLIP, oppure nome modello sentence-transformers
            device: 'cuda' o 'cpu' (auto-detect se None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        if model_name == 'clip':
            self._init_clip()
        else:
            self._init_sentence_transformer(model_name)
    
    def _init_clip(self):
        """Inizializza CLIP text encoder."""
        try:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            self.dim = 512  # CLIP ViT-B/32 output dim
            self.tokenize = clip.tokenize
            self._encoder_type = 'clip'
            print(f"Loaded CLIP ViT-B/32 text encoder (dim={self.dim})")
        except ImportError:
            print("CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
            print("Falling back to sentence-transformers...")
            self._init_sentence_transformer('all-MiniLM-L6-v2')
    
    def _init_sentence_transformer(self, model_name):
        """Inizializza sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(model_name)
            self.st_model.eval()
            self.dim = self.st_model.get_sentence_embedding_dimension()
            self._encoder_type = 'sentence_transformer'
            print(f"Loaded sentence-transformer {model_name} (dim={self.dim})")
        except ImportError:
            print("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
    
    def encode(self, texts):
        """
        Encoda una lista di testi.
        
        Args:
            texts: Lista di stringhe
            
        Returns:
            embeddings: np.array [N, dim]
        """
        if self._encoder_type == 'clip':
            return self._encode_clip(texts)
        else:
            return self._encode_st(texts)
    
    def _encode_clip(self, texts):
        """Encoding con CLIP."""
        with torch.no_grad():
            # CLIP ha limite di 77 token, tronca se necessario
            tokens = self.tokenize(texts, truncate=True).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            embeddings = embeddings.cpu().numpy().astype(np.float32)
        return embeddings
    
    def _encode_st(self, texts):
        """Encoding con sentence-transformers."""
        with torch.no_grad():
            embeddings = self.st_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def encode_task_graph(self, task_graph_json):
        """
        Encoda tutti i nodi di un task graph.
        
        Args:
            task_graph_json: dict con 'steps' (id -> description)
            
        Returns:
            node_ids: Lista di node_ids (ordinati)
            embeddings: np.array [N_nodes, dim]
        """
        steps = task_graph_json['steps']
        
        # Ordina per id (escludendo START e END)
        node_ids = []
        descriptions = []
        
        for node_id, desc in sorted(steps.items(), key=lambda x: int(x[0])):
            if desc not in ['START', 'END']:
                node_ids.append(int(node_id))
                # Pulizia descrizione (rimuovi prefisso action-action se presente)
                clean_desc = desc
                if '-' in desc and desc.split('-')[0].lower() == desc.split('-')[1].split()[0].lower():
                    # Es: "Chop-Chop 1 tsp cilantro" -> "Chop 1 tsp cilantro"
                    parts = desc.split('-', 1)
                    clean_desc = parts[1].strip()
                descriptions.append(clean_desc)
        
        embeddings = self.encode(descriptions)
        
        return node_ids, embeddings


if __name__ == "__main__":
    # Test del modello
    print("Testing TaskGraphMatcher...")
    
    # Test con configurazione perception/CLIP (default)
    model = TaskGraphMatcher(visual_dim=768, text_dim=512, hidden_dim=256)
    
    # Dummy input
    B, N_steps, N_nodes = 2, 10, 15
    visual_emb = torch.randn(B, N_steps, 768)
    text_emb = torch.randn(B, N_nodes, 512)
    
    prob, matching_cost = model(visual_emb, text_emb)
    
    print(f"Input: visual [{B}, {N_steps}, 768], text [{B}, {N_nodes}, 512]")
    print(f"Output prob shape: {prob.shape}")
    print(f"Matching cost: {matching_cost}")
    print(f"Probability: {prob.squeeze()}")
    
    print("\nTesting with Omnivore/sentence-transformer config...")
    model2 = TaskGraphMatcher(visual_dim=1024, text_dim=384, hidden_dim=256)
    visual_emb2 = torch.randn(B, N_steps, 1024)
    text_emb2 = torch.randn(B, N_nodes, 384)
    prob2, _ = model2(visual_emb2, text_emb2)
    print(f"Omnivore config output: {prob2.squeeze()}")
    
    print("\nTesting TextEncoder with CLIP...")
    try:
        encoder = TextEncoder('clip')
        test_texts = ["Chop the onion", "Add salt to the pan", "Stir the mixture"]
        emb = encoder.encode(test_texts)
        print(f"CLIP text embedding shape: {emb.shape}")
    except Exception as e:
        print(f"CLIP not available: {e}")
        print("Testing with sentence-transformers...")
        encoder = TextEncoder('all-MiniLM-L6-v2')
        emb = encoder.encode(test_texts)
        print(f"Sentence-transformer embedding shape: {emb.shape}")
    
    print("\nDone!")
