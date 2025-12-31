# Regex Generator

A desktop GUI tool for generating regex patterns from highlighted text using LLM assistance. Select text examples you want to match, optionally mark exclusions, and let the AI generate optimized regex patterns.

![AutoHotkey v2.0](https://img.shields.io/badge/AutoHotkey-v2.0-brightgreen)
![Python 3.x](https://img.shields.io/badge/Python-3.x-3776AB)
![Ollama](https://img.shields.io/badge/LLM-Ollama-orange)

## Features

- **Visual Text Selection**: Highlight text to create match examples (cyan) or exclusions (red)
- **LLM-Powered Generation**: Uses local Ollama with qwen2.5-coder:3b for intelligent pattern creation
- **Multi-Criteria Scoring**: 5-weight system balances accuracy, exclusions, coherence, generalization, and simplicity
- **Training Dataset**: 227+ curated examples with ability to add your own
- **Pre-compilation**: Optional rule extraction for faster runtime inference
- **Session Config**: Adjust model, temperature, and scoring weights on the fly

## Prerequisites

- **AutoHotkey v2.0** - [Download](https://www.autohotkey.com/)
- **Python 3.x** with packages:
  ```bash
  pip install dspy grex ollama
  ```
- **Ollama** running locally with model installed:
  ```bash
  ollama serve
  ollama pull qwen2.5-coder:3b
  ```

## Installation

1. Clone or download this repository
2. Install Python dependencies: `pip install dspy grex ollama`
3. Ensure Ollama is running: `ollama serve`
4. Pull the model: `ollama pull qwen2.5-coder:3b`

## Usage

### GUI Application

```bash
autohotkey regexgen.ahk
```

**Workflow:**
1. Paste or type text into the textbox
2. Highlight text you want to match (cyan highlight)
3. Right-click selections to mark as exclusions (red highlight)
4. Click "Generate" to create regex patterns
5. View results in the Results tab, copy patterns, or add to training dataset

### Command Line

```bash
# Run test suite
python regexgen.py --test

# Pre-compile for faster runtime
python regexgen.py --compile

# Generate regex from JSON input
python regexgen.py input.json output.json

# With custom config
python regexgen.py input.json output.json --config config.json

# Dataset management
python regexgen.py --list-dataset output.json
python regexgen.py --add-example example.json
python regexgen.py --delete-example <index>
```

## Project Structure

```
regen/
├── regexgen.ahk         # AutoHotkey host application
├── regexgen.py          # Python backend (DSPy + Ollama)
├── regexgen.html        # Web frontend
├── regexgen.js          # Frontend logic
├── regexgen.css         # Styles
├── lib/                 # AutoHotkey dependencies
│   ├── WebViewToo.ahk   # WebView2 wrapper
│   ├── WebView2.ahk     # WebView2 bindings
│   ├── JSON.ahk         # JSON parsing
│   ├── Promise.ahk      # Async support
│   ├── ComVar.ahk       # COM utilities
│   ├── 32bit/           # WebView2Loader.dll (32-bit)
│   └── 64bit/           # WebView2Loader.dll (64-bit)
└── dspy/                # Training data
    ├── regex-dspy-train.json   # Training examples
    ├── regex_compiled.json     # Pre-compiled program
    └── validate_trainset.py    # Validation script
```

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   AutoHotkey    │────▶│  Web Frontend   │────▶│     Python      │
│   (Host)        │◀────│  (WebView2)     │◀────│   (DSPy/LLM)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     Window              Text Selection          Regex Generation
     Management          Highlighting            Multi-criteria
     IPC Bridge          Results Display         Scoring
```

## Scoring System

Generated patterns are evaluated on 5 weighted criteria:

| Weight | Metric | Description |
|--------|--------|-------------|
| 35% | matches_all | Pattern matches all positive examples |
| 25% | excludes_all | Pattern avoids all negative examples |
| 15% | coherence | Extra matches are similar to examples |
| 15% | generalization | Uses character classes (`\d`, `\w`, etc.) |
| 10% | simplicity | Shorter patterns with less branching |

## Configuration

The Config tab allows session-level adjustments:

- **Model**: Ollama model name (default: `qwen2.5-coder:3b`)
- **Temperature**: LLM creativity (default: 0.7)
- **Max Attempts**: Refinement iterations (default: 10)
- **Reward Threshold**: Stop early if score exceeds (default: 0.85)
- **Scoring Weights**: Adjust the 5 criteria weights

## License

MIT License
