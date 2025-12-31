# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Regex Generator is a desktop GUI tool for generating regex patterns from highlighted text using LLM assistance. Users highlight text examples they want to match (positive examples) and right-click to mark exclusions (negative examples), then the tool generates regex patterns via a local Ollama LLM.

## Architecture

The application uses a WebView2-based GUI architecture with three layers:

1. **AutoHotkey Host** (`regespy.ahk`): Window management, clipboard monitoring, Python process orchestration
2. **Web Frontend** (`regespy.html/js/css`): Text selection UI, highlight visualization, results display
3. **Python Backend** (`regespy.py`): DSPy-based regex generation using Ollama LLM with multi-criteria scoring

### Data Flow
```
User pastes/types text in editable textbox -> JS stores as originalText
User highlights text -> JS calculates position -> Stores {text, start, end, negative}
User clicks "Generate" -> JS sends selections -> AHK writes temp JSON -> Python generates patterns
Python writes output JSON -> AHK reads -> JS displays patterns with copy buttons
```

### AHK-JS Communication
- AHK->JS: `MyWindow.ExecuteScriptAsync('functionName(args)')`
- JS->AHK: `ahk.functionName(args)` (registered via `AddCallbackToScript`)

Key JS functions called from AHK: `onRegexGenerated()`, `setLoading()`, `showError()`, `onDatasetLoaded()`, `onExampleAdded()`, `onDatasetItemDeleted()`
Key AHK functions called from JS: `generateRegex()`, `cancel()`, `loadDataset()`, `addToDataset()`, `deleteFromDataset()`

## Dependencies

- **AutoHotkey v2.0** with WebViewToo.ahk, JSON.ahk, qol_helper.ahk libraries (in `../../../Lib/`)
- **Python 3.x** with: `dspy`, `grex`, `ollama`
- **Ollama** running locally (`http://localhost:11434`) with model `qwen2.5:3b`

## Project Structure

```
regen/
├── regespy.ahk          # AutoHotkey host
├── regespy.py           # Python backend
├── regespy.html/js/css  # Web frontend
└── dspy/                 # DSPy training data
    ├── regex-dspy-train.json   # Training examples (227+)
    ├── regex_compiled.json     # Pre-compiled program
    └── validate_trainset.py    # Validation script
```

## Running

```bash
# Run the GUI application
autohotkey regespy.ahk

# Run Python test suite (detailed output with scores)
python regespy.py --test

# Pre-compile for faster runtime (saves to dspy/regex_compiled.json)
python regespy.py --compile

# Validate training set
python dspy/validate_trainset.py

# CLI usage (called by AHK)
python regespy.py <input.json> <output.json>

# CLI with custom config override
python regespy.py <input.json> <output.json> --config <config.json>

# Dataset management
python regespy.py --list-dataset <output.json>    # Export dataset
python regespy.py --add-example <example.json>    # Add example
python regespy.py --delete-example <index>        # Delete by index
```

## Python Regex Generation

The Python backend uses DSPy with `dspy.Refine` for iterative pattern improvement:

1. **Pattern Hints**: `analyze_match_items()` extracts observations (numeric, uppercase, wrappers, prefixes, etc.)
2. **Training Examples**: Loaded from `dspy/regex-dspy-train.json` plus grex-generated baseline
3. **Pre-compilation**: Optional `--compile` uses `InferRules` to extract natural language rules from training examples and add them to the prompt instructions. This helps the LLM learn patterns like "use `\d` for digits" from successful examples.
4. **5-Weight Scoring System**:
   - `matches_all` (0.35): Pattern matches all required items
   - `excludes_all` (0.25): Pattern avoids all excluded items
   - `coherence` (0.15): Extra matches similar to match_items (k-NN style pairwise similarity using length, char profile, Jaccard bigrams)
   - `generalization` (0.15): Uses character classes (`\d`, `\w`, `[A-Z]`)
   - `simplicity` (0.10): Shorter patterns with less branching preferred (cyclomatic complexity style)

   **Note**: When no exclude_items are provided, the `excludes_all` weight is redistributed proportionally to the other 4 metrics (no free points for untested constraints).
5. **Refinement Loop**: Up to 10 attempts with 0.85 reward threshold

Key configuration in `Config` dataclass:
- `model`: Ollama model name (default: `qwen2.5:3b`)
- `temperature`: LLM temperature (default: 0.4)
- `max_attempts`, `reward_threshold`: Refine parameters
- `compiled_program_path`: Path to pre-compiled program (default: `dspy/regex_compiled.json`)
- `compile_candidates`, `compile_num_rules`, `compile_threads`: InferRules optimizer params
- `weights`: Scoring weights dictionary

## Selection Mechanics

- Left-click drag: Select text (adds as positive/match example, cyan highlight)
- Right-click on selection: Toggle to negative/exclude (red highlight)
- Click on highlight: Remove selection
- Selections stored as `{text: string, start: number, end: number, negative: boolean}[]`

### Position-Based Selection (Important Implementation Detail)

The textbox is `contenteditable` and users can select the same text multiple times (e.g., selecting different "ABA" occurrences in "ABABABA"). The position calculation must handle:

1. **Overlapping patterns**: "ABABABA" contains "ABA" at positions 0, 2, and 4
2. **DOM structure changes**: After highlighting, the DOM has `<span>` elements mixed with text nodes
3. **Newline normalization**: Windows `\r\n` is normalized to `\n` to ensure consistent positions

**Algorithm** (`regespy.js` mouseup handler):
1. Create a Range from textbox start to selection start
2. Get `approxStart = beforeRange.toString().length` (text length before selection)
3. Find ALL occurrences of selected text in `originalText` using indexOf
4. Count occurrences where `start < approxStart` to determine which occurrence was selected
5. Pick `occurrences[countBefore]` as the target

This approach is robust because it counts occurrences that *start before* the selection point, correctly handling overlapping patterns where a match might partially overlap with the selection start.

**Key sync point**: After `highlightSelections()` sets innerHTML, `originalText` is updated to match `textbox.textContent` to handle any browser normalization.

## UI Tabs

The interface uses a 4-tab layout:

1. **Generate Tab**: Main text highlighting workflow for selecting match/exclude items
2. **Results Tab**: Sortable table showing generated patterns with:
   - Medal icons (gold/silver/bronze) for top 3 results by score
   - Comprehensive columns: Rank, Pattern, Score, Matches, Missed, Excluded, Source, Actions
   - Star button to add a result to the training dataset (shows confirmation modal)
   - Copy button for pattern
3. **Dataset Tab**: Manage training examples with:
   - Search/filter functionality
   - Truncated text with hover tooltips
   - Delete button for each example
   - Refresh button to reload from file
4. **Config Tab**: Session-level config editing (not persisted):
   - Model name, temperature, max attempts, reward threshold
   - Visual weight bars for scoring weights
   - Reset to defaults button
