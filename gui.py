"""
gui.py - Interfaz Gráfica de Escritorio (Desktop GUI) con Tkinter
Permite entrenar el Mini-Transformer y generar texto de forma interactiva con controles deslizantes.
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# Intentar importar PyTorch y módulos locales
try:
    import torch
    from model import MiniTransformerLM
    from generate import load_model_and_vocab, generate_completion, select_device
    import train
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class MiniTransformerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini-Transformer Generativo - Interfaz Interactiva")
        self.root.geometry("860x720")
        self.root.minsize(700, 600)

        self.model = None
        self.stoi = None
        self.itos = None
        self.device = None
        self.is_training = False

        self._apply_styles()
        self._build_ui()
        self._check_environment()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("TButton", font=("Helvetica", 11, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"))
        style.configure("SubHeader.TLabel", font=("Helvetica", 10, "italic"))

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Encabezado
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header_frame, text="🧠 Mini-Transformer Generativo", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(header_frame, text="Modelo de lenguaje a nivel de caracteres (Decoder-only Transformer)", style="SubHeader.TLabel").pack(anchor=tk.W)

        # Barra de estado superior
        self.status_var = tk.StringVar(value="Comprobando entorno...")
        status_lbl = ttk.Label(header_frame, textvariable=self.status_var, foreground="#0066cc", font=("Helvetica", 10, "bold"))
        status_lbl.pack(anchor=tk.E, pady=(2, 0))

        # Cuaderno de pestañas (Generación / Entrenamiento)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        # --- Pestaña 1: Generación ---
        gen_tab = ttk.Frame(notebook, padding="10")
        notebook.add(gen_tab, text=" Generación de Texto ")
        self._build_gen_tab(gen_tab)

        # --- Pestaña 2: Entrenamiento ---
        train_tab = ttk.Frame(notebook, padding="10")
        notebook.add(train_tab, text=" Entrenar Modelo ")
        self._build_train_tab(train_tab)

    def _build_gen_tab(self, parent):
        # Panel superior: Prompt
        prompt_frame = ttk.LabelFrame(parent, text="Texto inicial (Prompt)", padding="10")
        prompt_frame.pack(fill=tk.X, pady=(0, 10))

        self.prompt_entry = ttk.Entry(prompt_frame, font=("Helvetica", 12))
        self.prompt_entry.pack(fill=tk.X, pady=5)
        self.prompt_entry.insert(0, "En un lugar de la Mancha ")

        # Panel de Parámetros (Sliders)
        params_frame = ttk.LabelFrame(parent, text="Parámetros de Muestreo", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))

        # Grid de parámetros
        params_frame.columnconfigure(1, weight=1)

        # Temperatura
        ttk.Label(params_frame, text="Temperatura:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.temp_var = tk.DoubleVar(value=0.75)
        self.temp_slider = ttk.Scale(params_frame, from_=0.1, to=1.5, variable=self.temp_var, command=lambda v: self.temp_val_lbl.config(text=f"{float(v):.2f}"))
        self.temp_slider.grid(row=0, column=1, sticky=tk.EW, padx=10)
        self.temp_val_lbl = ttk.Label(params_frame, text="0.75", width=6)
        self.temp_val_lbl.grid(row=0, column=2, sticky=tk.W)

        # Top-K
        ttk.Label(params_frame, text="Top-K:").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.topk_var = tk.IntVar(value=15)
        self.topk_slider = ttk.Scale(params_frame, from_=1, to=50, variable=self.topk_var, command=lambda v: self.topk_val_lbl.config(text=str(int(float(v)))))
        self.topk_slider.grid(row=1, column=1, sticky=tk.EW, padx=10)
        self.topk_val_lbl = ttk.Label(params_frame, text="15", width=6)
        self.topk_val_lbl.grid(row=1, column=2, sticky=tk.W)

        # Caracteres máximos
        ttk.Label(params_frame, text="Longitud (Tokens):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.tokens_var = tk.IntVar(value=300)
        self.tokens_slider = ttk.Scale(params_frame, from_=50, to=800, variable=self.tokens_var, command=lambda v: self.tokens_val_lbl.config(text=str(int(float(v)))))
        self.tokens_slider.grid(row=2, column=1, sticky=tk.EW, padx=10)
        self.tokens_val_lbl = ttk.Label(params_frame, text="300", width=6)
        self.tokens_val_lbl.grid(row=2, column=2, sticky=tk.W)

        # Botón de Generación
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        self.gen_btn = ttk.Button(btn_frame, text="✨ Generar Texto", command=self._on_generate_click)
        self.gen_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.load_btn = ttk.Button(btn_frame, text="🔄 Recargar Modelo", command=self._load_model)
        self.load_btn.pack(side=tk.LEFT)

        # Área de Salida
        out_frame = ttk.LabelFrame(parent, text="Resultado Generado", padding="10")
        out_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(out_frame, wrap=tk.WORD, font=("Courier", 11), height=12)
        self.output_text.pack(fill=tk.BOTH, expand=True)

    def _build_train_tab(self, parent):
        info_frame = ttk.LabelFrame(parent, text="Configuración del Entrenamiento", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_frame.columnconfigure(1, weight=1)

        ttk.Label(info_frame, text="Iteraciones:").grid(row=0, column=0, sticky=tk.W, pady=4)
        self.iters_var = tk.IntVar(value=1000)
        iters_entry = ttk.Entry(info_frame, textvariable=self.iters_var, width=10)
        iters_entry.grid(row=0, column=1, sticky=tk.W, padx=10)

        ttk.Label(info_frame, text="Tamaño de lote (batch size):").grid(row=1, column=0, sticky=tk.W, pady=4)
        self.batch_var = tk.IntVar(value=32)
        batch_entry = ttk.Entry(info_frame, textvariable=self.batch_var, width=10)
        batch_entry.grid(row=1, column=1, sticky=tk.W, padx=10)

        ttk.Label(info_frame, text="Contexto (block size):").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.block_var = tk.IntVar(value=64)
        block_entry = ttk.Entry(info_frame, textvariable=self.block_var, width=10)
        block_entry.grid(row=2, column=1, sticky=tk.W, padx=10)

        # Botón Iniciar Entrenamiento
        self.train_btn = ttk.Button(info_frame, text="🚀 Iniciar Entrenamiento", command=self._on_train_click)
        self.train_btn.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Consola de Registro (Logs)
        log_frame = ttk.LabelFrame(parent, text="Progreso y Métricas", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Courier", 10), height=14, background="#1e1e1e", foreground="#d4d4d4")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _check_environment(self):
        if not TORCH_AVAILABLE:
            self.status_var.set("⚠️ PyTorch no detectado. Ejecuta ./setup.sh primero.")
            self.gen_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.DISABLED)
            messagebox.showwarning(
                "Dependencias no encontradas",
                "PyTorch no está instalado en el entorno de Python actual.\n\n"
                "Para activarlo:\n"
                "1. Abre la terminal en esta carpeta.\n"
                "2. Ejecuta './setup.sh' para crear el entorno e instalar PyTorch.\n"
                "3. Luego ejecuta 'python gui.py' dentro del entorno activado."
            )
            return

        self.device = select_device()
        self._load_model()

    def _load_model(self):
        model_path = "output/model.pt"
        vocab_path = "output/vocab.json"

        if os.path.exists(model_path) and os.path.exists(vocab_path):
            try:
                self.model, self.stoi, self.itos = load_model_and_vocab(model_path, vocab_path, self.device)
                self.status_var.set(f"✅ Modelo cargado ({self.device})")
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, "Modelo cargado y listo para generar texto.\nEscribe un prompt y presiona 'Generar Texto'.")
            except Exception as e:
                self.status_var.set("❌ Error al cargar modelo")
                messagebox.showerror("Error al cargar modelo", str(e))
        else:
            self.status_var.set(f"ℹ️ Sin entrenar ({self.device}) - Ve a la pestaña 'Entrenar Modelo'")
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Aún no se ha entrenado el modelo.\n\nPuedes ir a la pestaña 'Entrenar Modelo' y hacer clic en 'Iniciar Entrenamiento' para entrenarlo en pocos segundos.")

    def _on_generate_click(self):
        if self.model is None:
            messagebox.showinfo("Modelo no disponible", "Primero debes entrenar el modelo en la pestaña 'Entrenar Modelo'.")
            return

        prompt = self.prompt_entry.get().strip()
        if not prompt:
            prompt = " "

        temperature = self.temp_var.get()
        top_k = int(self.topk_var.get())
        max_tokens = int(self.tokens_var.get())

        self.gen_btn.config(state=tk.DISABLED, text="Generando...")
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"Generando texto a partir de: '{prompt}'...\n\n")

        def run_gen():
            try:
                result = generate_completion(
                    self.model, self.stoi, self.itos, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=self.device
                )
                self.root.after(0, lambda: self._show_generation_result(result))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error de Generación", str(e)))
            finally:
                self.root.after(0, lambda: self.gen_btn.config(state=tk.NORMAL, text="✨ Generar Texto"))

        threading.Thread(target=run_gen, daemon=True).start()

    def _show_generation_result(self, text):
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)

    def _on_train_click(self):
        if self.is_training:
            return

        self.is_training = True
        self.train_btn.config(state=tk.DISABLED, text="Entrenando...")
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "[*] Iniciando preparación de entrenamiento...\n")

        def log_msg(msg):
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)

        def run_train():
            try:
                data_path = "data/input.txt"
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"No se encontró {data_path}")

                with open(data_path, "r", encoding="utf-8") as f:
                    text = f.read()

                chars = sorted(list(set(text)))
                vocab_size = len(chars)
                stoi = {ch: i for i, ch in enumerate(chars)}
                data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

                n = int(0.9 * len(data))
                train_data = data[:n]
                val_data = data[n:]

                block_size = int(self.block_var.get())
                batch_size = int(self.batch_var.get())
                max_iters = int(self.iters_var.get())

                model = MiniTransformerLM(
                    vocab_size=vocab_size,
                    block_size=block_size,
                    n_embd=128,
                    n_head=4,
                    n_layer=3,
                    dropout=0.1
                ).to(self.device)

                total_params = sum(p.numel() for p in model.parameters())
                self.root.after(0, lambda: log_msg(f"[*] Dataset: {len(text):,} caracteres | Vocabulario: {vocab_size}"))
                self.root.after(0, lambda: log_msg(f"[*] Parámetros del modelo: {total_params:,}"))
                self.root.after(0, lambda: log_msg(f"[*] Dispositivo: {self.device}"))

                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

                def get_batch(split):
                    d = train_data if split == "train" else val_data
                    ix = torch.randint(len(d) - block_size, (batch_size,))
                    x = torch.stack([d[i:i + block_size] for i in ix]).to(self.device)
                    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix]).to(self.device)
                    return x, y

                for iter_idx in range(1, max_iters + 1):
                    xb, yb = get_batch("train")
                    logits, loss = model(xb, yb)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    if iter_idx % 100 == 0 or iter_idx == max_iters:
                        current_loss = loss.item()
                        msg = f"Iteración {iter_idx:4d}/{max_iters} | Pérdida actual: {current_loss:.4f}"
                        self.root.after(0, lambda m=msg: log_msg(m))

                # Guardar
                os.makedirs("output", exist_ok=True)
                vocab_meta = {
                    "chars": chars,
                    "vocab_size": vocab_size,
                    "block_size": block_size,
                    "n_embd": 128,
                    "n_head": 4,
                    "n_layer": 3,
                }
                with open("output/vocab.json", "w", encoding="utf-8") as f:
                    json.dump(vocab_meta, f, ensure_ascii=False, indent=2)

                torch.save(model.state_dict(), "output/model.pt")

                self.root.after(0, lambda: log_msg("\n[+] ¡Entrenamiento finalizado exitosamente!"))
                self.root.after(0, self._load_model)
                self.root.after(0, lambda: messagebox.showinfo("Entrenamiento Exitoso", "El modelo se ha entrenado y guardado correctamente.\n¡Ya puedes ir a la pestaña 'Generación de Texto'!"))
            except Exception as e:
                self.root.after(0, lambda err=str(e): log_msg(f"\n[!] Error: {err}"))
                self.root.after(0, lambda err=str(e): messagebox.showerror("Error en Entrenamiento", err))
            finally:
                self.is_training = False
                self.root.after(0, lambda: self.train_btn.config(state=tk.NORMAL, text="🚀 Iniciar Entrenamiento"))

        threading.Thread(target=run_train, daemon=True).start()


def main():
    root = tk.Tk()
    app = MiniTransformerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
