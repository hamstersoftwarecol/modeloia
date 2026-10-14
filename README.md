# Mini-Transformer Generativo a Nivel de Caracteres

Este proyecto contiene una implementación completa, modular y didáctica de un **modelo de lenguaje autorregresivo estilo GPT (Decoder-only Transformer)** diseñado para entrenarse y ejecutarse localmente, con **interfaces gráficas interactivas** (de escritorio y web).

---

## 📁 Estructura del Proyecto

```text
modeloia/
├── data/
│   └── input.txt          # Dataset de texto en español para entrenamiento
├── model.py               # Arquitectura del Transformer (Self-Attention, Bloques, LM Head)
├── train.py               # Pipeline de tokenización, entrenamiento y evaluación
├── generate.py            # Generación e inferencia por línea de comandos
├── gui.py                 # Interfaz Gráfica de Escritorio (Desktop GUI con Tkinter)
├── app_web.py             # Interfaz Gráfica Web interactiva (se abre en el navegador)
├── requirements.txt       # Dependencias del proyecto (PyTorch)
├── setup.sh               # Script automatizado para configurar el entorno e instalar paquetes
└── README.md              # Documentación y guía de uso
```

---

## 🧠 Arquitectura del Modelo

El modelo implementa los componentes esenciales de los Grandes Modelos de Lenguaje (LLMs) modernos:
- **Token & Position Embeddings:** Representación vectorial de caracteres y codificación de orden secuencial.
- **Causal Self-Attention:** Autoatención con máscara triangular inferior para impedir que las posiciones atiendan a tokens futuros.
- **Multi-Head Attention:** División de la atención en múltiples cabezas para aprender diferentes patrones lingüísticos de forma paralela.
- **Pre-LayerNorm & Conexiones Residuales:** Estabilidad numérica durante el entrenamiento con gradiente descendente.
- **FeedForward Network (MLP):** Expansión a 4x con función de activación no lineal GELU.
- **Weight Tying:** Compartición de pesos entre la matriz de embedding y la capa de proyección final para mayor eficiencia.

---

## 🚀 Instalación Inicial

Ejecuta el script de configuración automática desde la terminal:

```bash
cd /Users/admin/Documents/modeloia
chmod +x setup.sh
./setup.sh
```

*(Esto creará la carpeta `venv` e instalará PyTorch con aceleración nativa para Apple Silicon).*

---

## 🖥️ Interfaces Gráficas Interactivas

Puedes interactuar con el modelo de dos formas visuales muy cómodas:

### Opción A: Interfaz de Escritorio (Desktop Window)
Una ventana nativa en macOS que te permite entrenar y generar texto con barras deslizantes interactivas:

```bash
cd /Users/admin/Documents/modeloia
source venv/bin/activate
python gui.py
```

### Opción B: Interfaz Web en el Navegador
Un servidor local ultraligero que abre automáticamente una aplicación web moderna en Safari o Chrome (http://localhost:8000):

```bash
cd /Users/admin/Documents/modeloia
source venv/bin/activate
python app_web.py
```

Ambas interfaces permiten:
- Escribir tu propio *prompt* de texto.
- Ajustar en tiempo real la **Temperatura** (creatividad), **Top-K** y la **Longitud en tokens**.
- Iniciar el entrenamiento del modelo con un solo clic y ver el progreso en vivo.

---

## ⌨️ Uso por Línea de Comandos (CLI)

Si prefieres trabajar directamente desde la terminal:

### 1. Entrenar el Modelo
```bash
source venv/bin/activate
python train.py
```

### 2. Generar Texto
```bash
python generate.py --prompt "En un lugar de la Mancha "
```

#### Modo interactivo en terminal:
```bash
python generate.py --interactive
```

---

## 📝 Usar tu Propio Dataset

Para entrenar el modelo con tus propios textos (cuentos, código, artículos, etc.):
1. Reemplaza o amplía el contenido de `data/input.txt`.
2. Vuelve a entrenar el modelo desde la interfaz gráfica o con `python train.py`.
