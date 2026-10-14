"""
train.py - Script de entrenamiento para el Mini-Transformer
Entrena el modelo autorregresivo a nivel de caracteres y guarda los pesos del modelo.
"""

import os
import json
import argparse
import time
import torch
from model import MiniTransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar Mini-Transformer Generativo")
    parser.add_argument("--data_path", type=str, default="data/input.txt", help="Ruta al archivo de texto")
    parser.add_argument("--output_dir", type=str, default="output", help="Directorio para guardar modelo y vocabulario")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote")
    parser.add_argument("--block_size", type=int, default=64, help="Longitud de contexto (caracteres)")
    parser.add_argument("--n_embd", type=int, default=128, help="Dimensión de los embeddings")
    parser.add_argument("--n_head", type=int, default=4, help="Número de cabezas de atención")
    parser.add_argument("--n_layer", type=int, default=3, help="Número de capas Transformer")
    parser.add_argument("--dropout", type=float, default=0.1, help="Tasa de dropout")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Tasa de aprendizaje")
    parser.add_argument("--max_iters", type=int, default=1200, help="Iteraciones máximas de entrenamiento")
    parser.add_argument("--eval_interval", type=int, default=200, help="Intervalo de evaluación")
    parser.add_argument("--eval_iters", type=int, default=50, help="Lotes para calcular pérdida promedio")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_iters, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
            y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    device = select_device()
    print(f"[*] Dispositivo seleccionado: {device}")

    # 1. Cargar datos
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[*] Dataset cargado. Longitud total: {len(text):,} caracteres.")

    # 2. Construir vocabulario
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    print(f"[*] Tamaño del vocabulario: {vocab_size} caracteres únicos.")

    # Codificar texto completo a tensor de enteros
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # 3. Separar en conjunto de entrenamiento y validación (90% / 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 4. Inicializar modelo
    model = MiniTransformerLM(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Parámetros totales del modelo: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Función para obtener un lote
    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i:i + args.block_size] for i in ix]).to(device)
        y = torch.stack([d[i + 1:i + args.block_size + 1] for i in ix]).to(device)
        return x, y

    # 5. Bucle de entrenamiento
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[*] Iniciando entrenamiento ({args.max_iters} iteraciones)...")
    start_time = time.time()

    for iter_idx in range(1, args.max_iters + 1):
        if iter_idx % args.eval_interval == 0 or iter_idx == args.max_iters:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size, args.block_size, args.eval_iters, device
            )
            elapsed = time.time() - start_time
            print(f"Iteración {iter_idx:4d}/{args.max_iters} | Pérdida Train: {losses['train']:.4f} | Pérdida Val: {losses['val']:.4f} | Tiempo: {elapsed:.1f}s")

        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 6. Guardar vocabulario y pesos del modelo
    vocab_meta = {
        "chars": chars,
        "vocab_size": vocab_size,
        "block_size": args.block_size,
        "n_embd": args.n_embd,
        "n_head": args.n_head,
        "n_layer": args.n_layer,
    }
    vocab_file = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

    model_file = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_file)

    print(f"\n[+] Entrenamiento completado exitosamente.")
    print(f"[+] Vocabulario guardado en: {vocab_file}")
    print(f"[+] Pesos guardados en: {model_file}")

    # 7. Demostración de generación
    print("\n--- Demostración de Generación Rápida ---")
    prompt = "En un lugar "
    context = torch.tensor([[stoi.get(c, 0) for c in prompt]], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=150, temperature=0.8)
    generated_text = "".join([itos.get(i.item(), "") for i in generated[0]])
    print(generated_text)
    print("----------------------------------------\n")


if __name__ == "__main__":
    main()
