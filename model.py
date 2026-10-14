"""
model.py - Arquitectura de un Mini-Transformer Generativo (Decoder-only)
Inspirado en la arquitectura GPT para procesamiento y generación a nivel de caracteres.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """
    Atención causal de múltiples cabezas (Multi-Head Causal Self-Attention).
    Garantiza que cada token/carácter solo atienda a tokens pasados.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd debe ser divisible por n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        # Proyecciones lineales para Query, Key, Value
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Proyección de salida
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularización
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Máscara causal triangular inferior (buffer no entrenable)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch, Secuencia (Tiempo), Canales (Embedding)

        # Calcular Q, K, V en un solo paso y dividir
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=2)

        # Reorganizar dimensiones: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Producto punto escalado: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        # Aplicar máscara causal (no mirar al futuro)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Ponderar valores y recombinar cabezas
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Proyección de salida y conexión residual
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Red neuronal feedforward por posición con activación GELU."""
    def __init__(self, n_embd: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Bloque Transformer básico: Pre-LayerNorm + Autoatención + FeedForward."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniTransformerLM(nn.Module):
    """
    Modelo de Lenguaje Autorregresivo estilo GPT a nivel de caracteres.
    """
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size

        # Embeddings de caracteres y posiciones
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Bloques de Transformer
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )

        # Normalización final y cabezal de predicción
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Compartir pesos entre token embedding y linear projection (weight tying)
        self.tok_emb.weight = self.lm_head.weight

        # Inicialización de pesos
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.block_size, f"Longitud de secuencia {T} supera el tamaño de bloque {self.block_size}"

        # Suma de embedding de token y embedding de posición
        tok_emb = self.tok_emb(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos)  # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Pasar a través de los bloques del Transformer
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Calcular entropía cruzada entre predicciones y etiquetas
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """
        Genera nuevos tokens de forma autorregresiva condicionada en la secuencia inicial idx.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Recortar contexto a block_size si la secuencia es más larga
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # Predecir logits
            logits, _ = self(idx_cond)
            # Tomar el último paso de tiempo
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Filtrado top-k opcional
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Convertir a probabilidades con Softmax
            probs = F.softmax(logits, dim=-1)

            # Muestrear el siguiente índice
            idx_next = torch.multinomial(probs, num_samples=1)
            # Concatenar a la secuencia
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
