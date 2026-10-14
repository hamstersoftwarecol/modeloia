"""
generate.py - Script de inferencia y generación de texto
Carga el modelo entrenado y su vocabulario para generar texto a partir de un texto inicial (prompt).
"""

import os
import json
import argparse
import torch
from model import MiniTransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generar texto con Mini-Transformer")
    parser.add_argument("--model_path", type=str, default="output/model.pt", help="Ruta al archivo model.pt")
    parser.add_argument("--vocab_path", type=str, default="output/vocab.json", help="Ruta al archivo vocab.json")
    parser.add_argument("--prompt", type=str, default="El caballero ", help="Texto inicial para condicionar la generación")
    parser.add_argument("--max_tokens", type=int, default=300, help="Cantidad máxima de caracteres a generar")
    parser.add_argument("--temperature", type=float, default=0.75, help="Temperatura de muestreo (menor = más conservador, mayor = más variado)")
    parser.add_argument("--top_k", type=int, default=15, help="Límite top-k para filtrar tokens poco probables")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo en consola")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_vocab(model_path, vocab_path, device):
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"No se encontró el vocabulario en {vocab_path}. Primero ejecuta train.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontraron los pesos del modelo en {model_path}. Primero ejecuta train.py")

    with open(vocab_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    chars = meta["chars"]
    vocab_size = meta["vocab_size"]
    block_size = meta["block_size"]
    n_embd = meta["n_embd"]
    n_head = meta["n_head"]
    n_layer = meta["n_layer"]

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    model = MiniTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, stoi, itos


def generate_completion(model, stoi, itos, prompt, max_tokens, temperature, top_k, device):
    # Codificar prompt ignorando caracteres fuera del vocabulario
    encoded = [stoi[c] for c in prompt if c in stoi]
    if not encoded:
        encoded = [0]

    context = torch.tensor([encoded], dtype=torch.long, device=device)
    output = model.generate(
        context,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )

    generated_chars = [itos.get(idx.item(), "") for idx in output[0]]
    return "".join(generated_chars)


def main():
    args = parse_args()
    device = select_device()

    try:
        model, stoi, itos = load_model_and_vocab(args.model_path, args.vocab_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print("=" * 60)
    print(" Mini-Transformer Generativo - Inferencia de Texto")
    print("=" * 60)
    print(f"Dispositivo: {device} | Temperatura: {args.temperature} | Top-K: {args.top_k}\n")

    if args.interactive:
        print("Modo interactivo activado. Escribe tu texto y presiona Enter (o 'salir' para terminar):")
        while True:
            try:
                user_prompt = input("\nPrompt > ")
                if user_prompt.strip().lower() in ["salir", "exit", "quit"]:
                    break
                if not user_prompt.strip():
                    continue
                result = generate_completion(
                    model, stoi, itos, user_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device
                )
                print("-" * 50)
                print(result)
                print("-" * 50)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nSesión finalizada.")
    else:
        print(f"Prompt inicial: '{args.prompt}'")
        print("-" * 60)
        result = generate_completion(
            model, stoi, itos, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        print(result)
        print("-" * 60)


if __name__ == "__main__":
    main()
