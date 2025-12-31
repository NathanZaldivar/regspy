"""
Regex Generator - DSPy + Refine
================================

Generates regex patterns from example match/exclude strings using DSPy with
Ollama LLM. Uses a 5-weight reward system (matches_all, excludes_all,
coherence, generalization, simplicity) to iteratively refine patterns.

Supports pre-compilation for faster runtime inference.

Prerequisites:
    pip install dspy grex
    ollama serve

Usage:
    python regespy.py --test                              # Run test cases
    python regespy.py --compile                           # Pre-compile for faster runtime
    python regespy.py <input.json> <output.json>          # Generate regex
    python regespy.py <input.json> <output.json> --config <config.json>  # With custom config
    python regespy.py --list-dataset <output.json>        # Export training dataset
    python regespy.py --add-example <example.json>        # Add example to dataset
    python regespy.py --delete-example <index>            # Delete example from dataset
"""

import dspy
from dspy.teleprompt import LabeledFewShot, InferRules
import os
import re
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from grex import RegExpBuilder


# ============================================================================
# SCRIPT DIRECTORY - for resolving relative paths
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()


def resolve_path(relative_path: str) -> str:
    """Resolve a relative path to an absolute path based on script directory."""
    return str(SCRIPT_DIR / relative_path)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for the regex generator"""
    model: str = "qwen2.5-coder:3b"
    ollama_url: str = "http://localhost:11434"
    dataset_file: str = field(default_factory=lambda: resolve_path("dspy/regex-dspy-train.json"))

    # Refine parameters
    max_attempts: int = 10           # N for dspy.Refine
    reward_threshold: float = 0.85  # Stop if we hit this score
    fail_count: int = None          # None = keep trying until N

    # LM settings
    temperature: float = 0.7        # Balanced temp for Refine (lower = more deterministic)
    enable_cache: bool = False

    # Few-shot settings
    few_shot_k: int = 10
    use_cot: bool = True            # CoT helps with reasoning about patterns

    # Validation settings
    debug: bool = True

    # Pre-compiled program settings (InferRules optimizer)
    compiled_program_path: str = field(default_factory=lambda: resolve_path("dspy/regex_compiled.json"))
    compile_threads: int = 4
    compile_candidates: int = 16    # Number of candidate programs to evaluate
    compile_num_rules: int = 10     # Number of rules to extract from examples


    # Reward weights - 5-weight system
    weights: dict = field(default_factory=lambda: {
        'matches_all': 0.35,      # Must match all required items
        'excludes_all': 0.25,     # Must not match excluded items
        'coherence': 0.15,        # Extra matches should be similar to match_items
        'generalization': 0.15,   # Uses \d, \w, [A-Z] instead of explicit ranges
        'simplicity': 0.10,       # Shorter patterns with less branching preferred
    })


# ============================================================================
# PATTERN HINT GENERATOR - Analyze match items and suggest patterns
# ============================================================================

def analyze_match_items(match_items: list[str], exclude_items: list[str] = None) -> dict:
    """
    Analyze match items to generate pattern hints for the LLM.
    Returns a dict with observations and suggested pattern fragments.
    """
    exclude_items = exclude_items or []
    hints = {
        'observations': [],
        'suggested_fragments': [],
        'avoid': []
    }

    if not match_items:
        return hints

    # Check if all uppercase (only for purely alphabetic items)
    if all(item.isalpha() and item.isupper() for item in match_items):
        hints['observations'].append("All items are UPPERCASE")
        hints['suggested_fragments'].append("[A-Z]+")

    # Check if all lowercase (only for purely alphabetic items)
    elif all(item.isalpha() and item.islower() for item in match_items):
        hints['observations'].append("All items are lowercase")
        hints['suggested_fragments'].append("[a-z]+")

    # Check if all purely numeric (digits only)
    if all(item.isdigit() for item in match_items):
        hints['observations'].append("All items are numeric (digits only)")
        hints['suggested_fragments'].append(r"\d+")
    # Check if all alphanumeric (but not purely numeric)
    elif all(item.isalnum() for item in match_items):
        hints['observations'].append("All items are alphanumeric")
        hints['suggested_fragments'].append(r"\w+")

    # Check if all contain digits
    if all(any(c.isdigit() for c in item) for item in match_items):
        hints['observations'].append("All items contain digits")
        hints['suggested_fragments'].append(r"\d+")

    # ========================================
    # STRUCTURE DETECTION - word_sep_digits patterns
    # ========================================
    
    # Check for common structural patterns like word_digits, word-digits, etc.
    structure_patterns = [
        (r'^[a-zA-Z]+_\d+$', r'\w+_\d+', 'word_digits with underscore'),
        (r'^[a-zA-Z]+-\d+$', r'\w+-\d+', 'word-digits with hyphen'),
        (r'^[a-zA-Z]+\.\d+$', r'\w+\.\d+', 'word.digits with dot'),
        (r'^[a-zA-Z]+:\d+$', r'\w+:\d+', 'word:digits with colon'),
        (r'^\d+_[a-zA-Z]+$', r'\d+_\w+', 'digits_word with underscore'),
        (r'^\d+-[a-zA-Z]+$', r'\d+-\w+', 'digits-word with hyphen'),
        (r'^[A-Z]\d+$', r'[A-Z]\d+', 'single letter followed by digits'),
        (r'^[A-Z]{1,3}\d+$', r'[A-Z]{1,3}\d+', 'short prefix followed by digits'),
        (r'^[a-z]+@[a-z]+\.[a-z]+$', r'[a-z]+@[a-z]+\.[a-z]+', 'email-like pattern'),
    ]
    
    for pattern, suggestion, description in structure_patterns:
        if all(re.match(pattern, item, re.IGNORECASE) for item in match_items):
            hints['observations'].append(f"All items follow '{description}' structure")
            hints['suggested_fragments'].insert(0, suggestion)  # High priority suggestion
            break  # Only add one structure suggestion

    # Check for common prefix
    if len(match_items) > 1:
        prefix = common_prefix(match_items)
        if len(prefix) >= 1 and not prefix.isspace():
            hints['observations'].append(f"Common prefix: '{prefix}'")
            # Only suggest literal prefix if it's short
            if len(prefix) <= 3:
                hints['suggested_fragments'].append(f"{re.escape(prefix)}")

    # Check for common suffix
    if len(match_items) > 1:
        suffix = common_suffix(match_items)
        if len(suffix) >= 1 and not suffix.isspace():
            hints['observations'].append(f"Common suffix: '{suffix}'")

    # Detect wrapper patterns (common prefix + suffix with special chars on both ends)
    # e.g., <|im_start|>, {{foo}}, [bar]
    if len(match_items) > 1:
        prefix = common_prefix(match_items)
        suffix = common_suffix(match_items)
        special_chars = set(r'[](){}|^$.*+?\/<>')
        has_special_prefix = prefix and any(c in special_chars for c in prefix)
        has_special_suffix = suffix and any(c in special_chars for c in suffix)

        # Require both prefix and suffix to have special chars (true wrapper pattern)
        if has_special_prefix and has_special_suffix:
            escaped_prefix = re.escape(prefix)
            escaped_suffix = re.escape(suffix)
            wrapper_pattern = f"{escaped_prefix}.+{escaped_suffix}"
            hints['observations'].append(f"Items wrapped in '{prefix}...{suffix}'")
            hints['suggested_fragments'].insert(0, wrapper_pattern)  # High priority

    # Check for common separators
    separators = set()
    for item in match_items:
        for sep in ['-', '_', '.', ':', '/']:
            if sep in item:
                separators.add(sep)

    if separators:
        hints['observations'].append(f"Contains separators: {separators}")

    # Check for consistent length
    lengths = [len(item) for item in match_items]
    if len(set(lengths)) == 1:
        hints['observations'].append(f"All items have length {lengths[0]}")

    # Check for patterns that would match excludes (to avoid)
    if exclude_items:
        # Check length differences
        match_lengths = set(len(item) for item in match_items)
        exclude_lengths = set(len(item) for item in exclude_items)
        
        if match_lengths.isdisjoint(exclude_lengths):
            hints['observations'].append(
                f"Match items have different lengths than excludes"
            )
            
        # Suggest word boundaries if excludes are superstrings (NOT lookahead - simpler is better)
        for exc in exclude_items:
            for match in match_items:
                if match in exc and match != exc:
                    hints['observations'].append(
                        f"'{match}' is substring of excluded '{exc}' - use \\b word boundaries to prevent partial matches"
                    )
                    hints['suggested_fragments'].append(r"\b")
                    hints['avoid'].append("Lookahead (?!) - use word boundaries \\b instead")
                    hints['avoid'].append("Patterns without word boundaries that match substrings")
                    break
        
        # Check if excludes are longer versions (suggests length-based distinction)
        match_lens = set(len(m) for m in match_items)
        exclude_lens = set(len(e) for e in exclude_items)
        if match_lens and exclude_lens and max(match_lens) < min(exclude_lens):
            hints['observations'].append(
                f"All matches are shorter than excludes - length-limited quantifiers like {{3}} may help"
            )

    return hints


def common_prefix(strings: list[str]) -> str:
    """Find common prefix of all strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def common_suffix(strings: list[str]) -> str:
    """Find common suffix of all strings."""
    if not strings:
        return ""
    suffix = strings[0]
    for s in strings[1:]:
        while not s.endswith(suffix):
            suffix = suffix[1:]
            if not suffix:
                return ""
    return suffix


def format_hints_for_prompt(hints: dict) -> str:
    """Format hints dict into a string for the LLM prompt."""
    parts = []
    
    if hints['observations']:
        parts.append("Observations: " + "; ".join(hints['observations']))
    
    if hints['suggested_fragments']:
        parts.append("Suggested fragments: " + ", ".join(hints['suggested_fragments']))
    
    if hints['avoid']:
        parts.append("Avoid: " + "; ".join(hints['avoid']))
    
    return " | ".join(parts) if parts else "No specific patterns detected"


# ============================================================================
# DSPY SIGNATURE
# ============================================================================

class GenerateRegex(dspy.Signature):
    """Generate a regex pattern matching all match_items but none of exclude_items.
    Use character classes (\\d, \\w, [A-Z]) not literals. No anchors (^$). Use \\b for word boundaries.
    """

    text: str = dspy.InputField(
        desc="The full text to search within using re.findall()"
    )
    match_items: list[str] = dspy.InputField(
        desc="Strings the pattern MUST match"
    )
    exclude_items: list[str] = dspy.InputField(
        desc="Strings the pattern must NOT match"
    )
    pattern_hints: str = dspy.InputField(
        desc="Analysis hints about the match items (character types, common patterns, etc.)"
    )

    pattern: str = dspy.OutputField(
        desc="Regex pattern using character classes [A-Z] \\d \\w NOT literal match strings"
    )


# ============================================================================
# SAFE UNESCAPE FUNCTION
# ============================================================================

def safe_unescape(pattern: str) -> str:
    """
    Safely unescape double-escaped regex metacharacters from LLM output.
    Only unescapes known regex sequences to avoid breaking valid patterns.
    """
    if not pattern or '\\\\' not in pattern:
        return pattern

    # Only unescape recognized regex metacharacter sequences
    # e.g., \\\\d -> \\d, \\\\w -> \\w, etc.
    metachar_escapes = ['d', 'D', 'w', 'W', 's', 'S', 'b', 'B', 'n', 't', 'r', 'A', 'Z']
    result = pattern
    for char in metachar_escapes:
        result = result.replace(f'\\\\{char}', f'\\{char}')

    return result


# ============================================================================
# REWARD FUNCTION
# ============================================================================

class PatternAnalysis:
    """Analyze a regex pattern for quality metrics."""

    def __init__(self, pattern: str, text: str, match_items: list[str], exclude_items: list[str]):
        self.raw_pattern = pattern
        self.pattern = safe_unescape(pattern)
        self.text = text
        self.match_items = match_items
        self.exclude_items = exclude_items
        self.matches = set()
        self.is_valid = False
        self._analyze()

    def _analyze(self):
        """Run the pattern and collect matches."""
        if not self.pattern:
            return

        try:
            self.matches = set(re.findall(self.pattern, self.text))
            self.is_valid = True
        except re.error:
            self.is_valid = False

    def matches_all_required(self) -> float:
        """Score: Does it match all required items? (0.0 to 1.0)"""
        if not self.is_valid or not self.match_items:
            return 0.0
        matched = sum(1 for item in self.match_items if item in self.matches)
        return matched / len(self.match_items)

    def excludes_all_forbidden(self) -> float:
        """Score: Does it avoid all excluded items? (0.0 to 1.0)"""
        if not self.is_valid:
            return 0.0
        if not self.exclude_items:
            return 1.0
        excluded_matches = self.matches & set(self.exclude_items)
        if not excluded_matches:
            return 1.0
        return 1.0 - (len(excluded_matches) / len(self.exclude_items))

    def no_hardcoded_strings(self) -> float:
        """Score: Are match_items NOT hardcoded in pattern? (0.0 to 1.0)"""
        if not self.pattern:
            return 0.0
        
        hardcoded_count = 0
        for item in self.match_items:
            # Check if the literal item appears in the pattern
            # But ignore if it's just a single char that's part of a class
            if len(item) > 2 and item.lower() in self.pattern.lower():
                hardcoded_count += 1
            elif len(item) <= 2:
                # For short items, check if they appear as literals (not in char class)
                # This is a simplified check
                if re.search(rf'(?<!\[){re.escape(item)}(?!\])', self.pattern, re.IGNORECASE):
                    hardcoded_count += 1
        
        if hardcoded_count == 0:
            return 1.0
        return max(0.0, 1.0 - (hardcoded_count / len(self.match_items)))

    def uses_character_classes(self) -> float:
        """Score: Does it use generic char classes? (0.0 to 1.0)"""
        if not self.pattern:
            return 0.0

        score = 0.0
        
        # Check for metacharacters (these appear as \d, \w, etc in the pattern string)
        metachar_patterns = [
            r'\\d',  # digit
            r'\\w',  # word char
            r'\\s',  # whitespace
            r'\\b',  # word boundary
            r'\\D',  # non-digit
            r'\\W',  # non-word
            r'\\S',  # non-whitespace
        ]
        
        for mp in metachar_patterns:
            if re.search(mp, self.pattern):
                score += 0.2
        
        # Check for bracket character classes like [A-Z], [a-z], [0-9]
        bracket_classes = [
            r'\[A-Z\]',
            r'\[a-z\]', 
            r'\[0-9\]',
            r'\[A-Za-z\]',
            r'\[\w-\]+',  # generic char class with range
        ]
        
        for bc in bracket_classes:
            if re.search(bc, self.pattern, re.IGNORECASE):
                score += 0.25
        
        # Bonus for quantifiers on classes (shows generalization)
        if re.search(r'(?:\\[dwsb]|\[[^\]]+\])[+*?]|\{[\d,]+\}', self.pattern):
            score += 0.15

        return min(1.0, score)

    def simplicity_score(self) -> float:
        """Score: Simpler patterns are better (0.0 to 1.0)

        Combines two factors:
        - Length score: Shorter patterns preferred (but not too short)
        - Complexity score: Fewer branches (|) and groups (()) preferred

        Based on cyclomatic complexity concept adapted for regex.
        """
        if not self.pattern:
            return 0.0

        # 1. Length score (0.0 to 1.0)
        length = len(self.pattern)
        if length < 5:
            length_score = 0.3  # Too short is suspicious
        elif length <= 20:
            length_score = 1.0
        elif length <= 40:
            length_score = 0.8
        elif length <= 60:
            length_score = 0.6
        elif length <= 100:
            length_score = 0.4
        else:
            length_score = 0.2

        # 2. Branching complexity score (0.0 to 1.0)
        # Count alternations and groups (similar to cyclomatic complexity)
        alternations = self.pattern.count('|')
        groups = self.pattern.count('(')

        # Complexity = 1 + branches + groups (cyclomatic-style)
        complexity = 1 + alternations + groups

        # Score: 1-2 = perfect, 3-4 = good, 5-7 = ok, 8+ = complex
        if complexity <= 2:
            complexity_score = 1.0
        elif complexity <= 4:
            complexity_score = 0.8
        elif complexity <= 7:
            complexity_score = 0.6
        else:
            complexity_score = 0.4

        # Weighted combination: length matters slightly more than complexity
        return 0.6 * length_score + 0.4 * complexity_score

    def coherence_score(self) -> float:
        """Score: Are extra matches similar to intended match_items? (0.0 to 1.0)

        Uses average pairwise similarity (k-NN style) - works with as few as 1 reference item.
        For each extra match, finds its max similarity to any reference item,
        then returns the mean coherence across all extra matches.

        Similarity is computed using:
        - Length similarity: 1 - |len(a) - len(b)| / max(len(a), len(b))
        - Character profile similarity: matching char types (upper, lower, digit, special)
        - Jaccard similarity on character bigrams (structural similarity)

        Example: If match_items are "TEST" and "EAT" (len 3-4, uppercase),
        then extra match "LOVE" (len 4, uppercase) has high similarity,
        but extra match "T" (len 1) has low similarity.
        """
        if not self.matches or not self.match_items:
            return 0.0

        wanted = set(self.match_items)
        excluded = set(self.exclude_items) if self.exclude_items else set()
        extra = self.matches - wanted - excluded

        # No extra matches = perfect coherence
        if not extra:
            return 1.0

        def get_char_profile(s):
            """Returns tuple of (has_upper, has_lower, has_digit, has_special)"""
            return (
                any(c.isupper() for c in s),
                any(c.islower() for c in s),
                any(c.isdigit() for c in s),
                any(not c.isalnum() for c in s)
            )

        def get_bigrams(s):
            """Get set of character bigrams for Jaccard similarity"""
            if len(s) < 2:
                return {s} if s else set()
            return set(s[i:i+2] for i in range(len(s) - 1))

        def pairwise_similarity(a, b):
            """Compute similarity between two strings (0.0 to 1.0)"""
            if not a or not b:
                return 0.0

            # 1. Length similarity (0 to 1)
            max_len = max(len(a), len(b))
            len_sim = 1.0 - abs(len(a) - len(b)) / max_len

            # 2. Character profile similarity (0 to 1)
            profile_a = get_char_profile(a)
            profile_b = get_char_profile(b)
            profile_sim = sum(pa == pb for pa, pb in zip(profile_a, profile_b)) / 4.0

            # 3. Jaccard similarity on bigrams (0 to 1)
            bigrams_a = get_bigrams(a)
            bigrams_b = get_bigrams(b)
            if bigrams_a and bigrams_b:
                jaccard = len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)
            else:
                jaccard = 1.0 if a == b else 0.0

            # Weighted average: length and profile matter more than exact structure
            return 0.35 * len_sim + 0.40 * profile_sim + 0.25 * jaccard

        # For each extra match, find max similarity to any reference item
        coherence_scores = []
        for item in extra:
            max_sim = max(pairwise_similarity(item, ref) for ref in wanted)
            coherence_scores.append(max_sim)

        # Return mean coherence across all extra matches
        return sum(coherence_scores) / len(coherence_scores)


def compute_reward(args: dict, pred: dspy.Prediction, config: Config) -> float:
    """Multi-criteria reward function for dspy.Refine."""
    weights = config.weights
    pattern = getattr(pred, 'pattern', '')
    exclude_items = args.get('exclude_items', [])

    analysis = PatternAnalysis(
        pattern=pattern,
        text=args.get('text', ''),
        match_items=args.get('match_items', []),
        exclude_items=exclude_items
    )

    # Invalid syntax = no score
    if not analysis.is_valid:
        if config.debug:
            print(f"    Pattern: {pattern}")
            print(f"    Invalid syntax - score: 0.0")
        return 0.0

    scores = {
        'matches_all': analysis.matches_all_required(),
        'excludes_all': analysis.excludes_all_forbidden(),
        'coherence': analysis.coherence_score(),
        'generalization': analysis.uses_character_classes(),
        'simplicity': analysis.simplicity_score(),
    }

    # Redistribute weights when no exclude_items provided
    # (don't give free points for an untested metric)
    if not exclude_items:
        active_weights = {k: v for k, v in weights.items() if k != 'excludes_all'}
        weight_sum = sum(active_weights.values())
        total_score = sum(scores[k] * active_weights[k] / weight_sum for k in active_weights)
    else:
        total_score = sum(scores[k] * weights[k] for k in scores)

    if config.debug:
        print(f"    Pattern: {pattern}")
        print(f"    Scores: {', '.join(f'{k}={v:.2f}' for k, v in scores.items())}")
        print(f"    Total: {total_score:.3f}")

    return total_score


# ============================================================================
# PATTERN RESULT - Detailed scoring for GUI display
# ============================================================================

@dataclass
class PatternResult:
    """Holds detailed scoring info for a single pattern."""
    pattern: str
    source: str  # 'llm' or 'grex'
    is_valid: bool
    total_score: float

    # Individual weight scores (0.0 to 1.0 each)
    scores: dict  # matches_all, excludes_all, coherence, generalization, simplicity

    # Match details
    all_matches: list[str]  # Everything the pattern matched in the text
    matched_items: list[str]  # Which match_items it successfully matched
    missed_items: list[str]  # Which match_items it failed to match
    excluded_matched: list[str]  # Which exclude_items it incorrectly matched (bad)
    extra_matches: list[str]  # Matches not in match_items or exclude_items


def score_pattern(pattern: str, text: str, match_items: list[str],
                  exclude_items: list[str], config: Config, source: str = "llm") -> PatternResult:
    """
    Score a pattern and return detailed results.
    Used for both LLM-generated and grex patterns.
    """
    weights = config.weights
    exclude_items = exclude_items or []

    # Run analysis
    analysis = PatternAnalysis(pattern, text, match_items, exclude_items)

    if not analysis.is_valid:
        return PatternResult(
            pattern=pattern,
            source=source,
            is_valid=False,
            total_score=0.0,
            scores={k: 0.0 for k in weights.keys()},
            all_matches=[],
            matched_items=[],
            missed_items=list(match_items),
            excluded_matched=[],
            extra_matches=[]
        )

    # Compute individual scores
    scores = {
        'matches_all': analysis.matches_all_required(),
        'excludes_all': analysis.excludes_all_forbidden(),
        'coherence': analysis.coherence_score(),
        'generalization': analysis.uses_character_classes(),
        'simplicity': analysis.simplicity_score(),
    }

    # Compute total with weight redistribution for empty exclude_items
    if not exclude_items:
        active_weights = {k: v for k, v in weights.items() if k != 'excludes_all'}
        weight_sum = sum(active_weights.values())
        total_score = sum(scores[k] * active_weights[k] / weight_sum for k in active_weights)
    else:
        total_score = sum(scores[k] * weights[k] for k in scores)

    # Categorize matches
    all_matches = list(analysis.matches)
    wanted = set(match_items)
    excluded = set(exclude_items)

    matched_items = [item for item in match_items if item in analysis.matches]
    missed_items = [item for item in match_items if item not in analysis.matches]
    excluded_matched = [item for item in exclude_items if item in analysis.matches]
    extra_matches = [m for m in analysis.matches if m not in wanted and m not in excluded]

    return PatternResult(
        pattern=pattern,
        source=source,
        is_valid=True,
        total_score=total_score,
        scores=scores,
        all_matches=all_matches,
        matched_items=matched_items,
        missed_items=missed_items,
        excluded_matched=excluded_matched,
        extra_matches=extra_matches
    )


def generate_grex_pattern(match_items: list[str]) -> str | None:
    """Generate a baseline pattern using grex."""
    try:
        return (
            RegExpBuilder.from_test_cases(match_items)
            .with_conversion_of_digits()
            .with_conversion_of_words()
            .with_conversion_of_repetitions()
            .without_anchors()
            .build()
        )
    except Exception:
        return None


# ============================================================================
# TRAINSET LOADER
# ============================================================================

def load_trainset(config: Config) -> list[dspy.Example]:
    """Load training examples from file."""

    trainset = []

    # Load from file
    try:
        with open(config.dataset_file, 'r') as f:
            data = json.load(f)

        for item in data[:config.few_shot_k]:
            hints = analyze_match_items(item["match_items"], item.get("exclude_items", []))
            example = dspy.Example(
                text=item["text"],
                match_items=item["match_items"],
                exclude_items=item.get("exclude_items", []),
                pattern_hints=format_hints_for_prompt(hints),
                pattern=item["expected_pattern"]
            ).with_inputs("text", "match_items", "exclude_items", "pattern_hints")
            trainset.append(example)
    except FileNotFoundError:
        if config.debug:
            print(f"[WARN] Dataset file not found: {config.dataset_file}")

    # NOTE: grex patterns are NOT added to training examples.
    # Grex is only used for scoring comparison, not as LLM training input.
    # This avoids teaching the LLM potentially flawed patterns (e.g., alternation
    # ordering issues where \w{5}|\w{7} matches "GOODB" instead of "GOODBYE").

    return trainset


# ============================================================================
# PRE-COMPILATION - One-time optimization for faster runtime
# ============================================================================

def compile_and_save(config: Config = None) -> str:
    """
    Pre-compile the regex generator using InferRules.

    InferRules extracts natural language rules from training examples and
    appends them to the prompt instructions. This helps the LLM learn patterns
    like "use \\d for digits" from successful examples.

    Run this once, then load the saved program for faster inference.

    Returns the path to the saved compiled program.
    """
    config = config or Config()

    print("[COMPILE] Setting up DSPy...")
    lm = dspy.LM(
        f'ollama_chat/{config.model}',
        api_base=config.ollama_url,
        api_key='',
        cache=True,  # Enable cache during compilation
        temperature=config.temperature
    )
    dspy.configure(lm=lm)

    # Load full training set (no dynamic grex examples during compilation)
    print("[COMPILE] Loading training data...")
    trainset = []
    try:
        with open(config.dataset_file, 'r') as f:
            data = json.load(f)
        for item in data:
            hints = analyze_match_items(item["match_items"], item.get("exclude_items", []))
            example = dspy.Example(
                text=item["text"],
                match_items=item["match_items"],
                exclude_items=item.get("exclude_items", []),
                pattern_hints=format_hints_for_prompt(hints),
                pattern=item["expected_pattern"]
            ).with_inputs("text", "match_items", "exclude_items", "pattern_hints")
            trainset.append(example)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found: {config.dataset_file}")
        return None

    print(f"[COMPILE] Loaded {len(trainset)} training examples")

    # Build base module
    if config.use_cot:
        base_module = dspy.ChainOfThought(GenerateRegex)
    else:
        base_module = dspy.Predict(GenerateRegex)

    # Reward function for optimization
    def metric_fn(example, pred, trace=None):
        args = {
            'text': example.text,
            'match_items': example.match_items,
            'exclude_items': example.exclude_items
        }
        return compute_reward(args, pred, config)

    # Run optimization with InferRules
    # InferRules extracts patterns from examples and adds them as explicit rules
    print(f"[COMPILE] Running InferRules...")
    print(f"  Candidates: {config.compile_candidates}")
    print(f"  Rules to extract: {config.compile_num_rules}")
    print(f"  Threads: {config.compile_threads}")

    optimizer = InferRules(
        metric=metric_fn,
        num_candidates=config.compile_candidates,
        num_rules=config.compile_num_rules,
        num_threads=config.compile_threads
    )

    compiled_program = optimizer.compile(
        student=base_module,
        trainset=trainset[:config.few_shot_k]  # Use subset for speed
    )

    # Save compiled program
    save_path = config.compiled_program_path
    compiled_program.save(save_path)
    print(f"[COMPILE] Saved compiled program to: {save_path}")

    return save_path


def load_compiled_program(config: Config):
    """Load a pre-compiled program if it exists."""
    if not os.path.exists(config.compiled_program_path):
        return None

    if config.use_cot:
        program = dspy.ChainOfThought(GenerateRegex)
    else:
        program = dspy.Predict(GenerateRegex)

    try:
        program.load(config.compiled_program_path)
        if config.debug:
            print(f"[LOADED] Pre-compiled program from {config.compiled_program_path}")
        return program
    except Exception as e:
        if config.debug:
            print(f"[WARN] Failed to load compiled program: {e}")
        return None


# ============================================================================
# MAIN GENERATOR with dspy.Refine
# ============================================================================

def generate_regex(input_data: dict, config: Config = None) -> dict:
    """
    Main entry point for regex generation using dspy.Refine.

    Returns a dict with:
        - results: list of PatternResult-like dicts (sorted by score, best first)
        - hints_used: pattern hints string
    """

    config = config or Config()

    text = input_data.get("text", "")
    match_items = input_data.get("Highlighted Items", [])
    exclude_items = input_data.get("Excluded Items", [])

    if not text or not match_items:
        raise ValueError("Input must contain 'text' and 'Highlighted Items'")

    # Generate hints for this specific input
    hints = analyze_match_items(match_items, exclude_items)
    hints_str = format_hints_for_prompt(hints)

    if config.debug:
        print(f"\n[PATTERN HINTS] {hints_str}")

    # Collect all pattern results
    pattern_results: list[PatternResult] = []

    # =========================
    # 1. Generate grex baseline
    # =========================
    grex_pattern = generate_grex_pattern(match_items)
    if grex_pattern:
        grex_result = score_pattern(grex_pattern, text, match_items, exclude_items, config, source="grex")
        pattern_results.append(grex_result)
        if config.debug:
            print(f"\n[GREX] {grex_pattern} (score: {grex_result.total_score:.3f})")

    # =========================
    # 2. Generate LLM pattern
    # =========================
    # Setup DSPy with Ollama
    lm = dspy.LM(
        f'ollama_chat/{config.model}',
        api_base=config.ollama_url,
        api_key='',
        cache=config.enable_cache,
        temperature=config.temperature
    )
    dspy.configure(lm=lm)

    # Try to load pre-compiled program first (faster)
    compiled_module = load_compiled_program(config)

    if compiled_module is None:
        # Fall back to runtime compilation
        trainset = load_trainset(config)

        # Build base module
        if config.use_cot:
            base_module = dspy.ChainOfThought(GenerateRegex)
        else:
            base_module = dspy.Predict(GenerateRegex)

        # Compile with few-shot if available
        if trainset:
            optimizer = LabeledFewShot(k=min(len(trainset), config.few_shot_k))
            compiled_module = optimizer.compile(student=base_module, trainset=trainset)
        else:
            compiled_module = base_module

    if config.debug:
        print(f"\n[GENERATING PATTERNS]")
        print(f"  Match: {match_items}")
        print(f"  Exclude: {exclude_items}")
        print(f"  Max attempts: {config.max_attempts}")
        print(f"  Stop threshold: {config.reward_threshold}")

    # Run multiple LLM generations to collect diverse patterns
    # Instead of relying on Refine's single output, we run the module multiple times
    # Duplicates don't count as attempts - LLM gets "free passes" to try again
    # Safety cap prevents infinite loops if LLM is completely stuck
    seen_llm_patterns = set()
    unique_attempts = 0
    total_calls = 0
    max_total_calls = config.max_attempts * 3  # Safety cap: 3x max attempts

    while unique_attempts < config.max_attempts and total_calls < max_total_calls:
        total_calls += 1
        try:
            result = compiled_module(
                text=text,
                match_items=match_items,
                exclude_items=exclude_items,
                pattern_hints=hints_str
            )

            pattern = safe_unescape(result.pattern)

            # Duplicate? Don't count as an attempt, just try again
            if pattern in seen_llm_patterns:
                if config.debug:
                    print(f"  (duplicate: {pattern}, retrying...)")
                continue

            # New unique pattern - count this attempt
            unique_attempts += 1
            seen_llm_patterns.add(pattern)
            llm_result = score_pattern(pattern, text, match_items, exclude_items, config, source="llm")
            pattern_results.append(llm_result)

            if config.debug:
                print(f"\n[LLM Attempt {unique_attempts}]")
                print(f"  Pattern: {pattern}")
                print(f"  Score: {llm_result.total_score:.3f}")

            # Early exit if we found a perfect pattern
            if llm_result.total_score >= config.reward_threshold:
                if config.debug:
                    print(f"  Hit threshold {config.reward_threshold}, stopping early")
                break

        except Exception as e:
            if config.debug:
                print(f"\n[LLM Attempt {unique_attempts + 1} ERROR] {e}")
            continue

    if config.debug and total_calls >= max_total_calls:
        print(f"\n  (hit safety cap of {max_total_calls} total calls, LLM may be stuck)")

    # Sort by score (best first), deduplicate by (pattern, source)
    # This keeps both LLM and grex even if they produce the same pattern
    seen_pattern_source = set()
    unique_results = []
    for r in sorted(pattern_results, key=lambda x: x.total_score, reverse=True):
        key = (r.pattern, r.source)
        if key not in seen_pattern_source:
            seen_pattern_source.add(key)
            unique_results.append(r)

    # Convert PatternResult objects to dicts for JSON serialization
    def result_to_dict(r: PatternResult) -> dict:
        return {
            "pattern": r.pattern,
            "source": r.source,
            "is_valid": r.is_valid,
            "total_score": round(r.total_score, 4),
            "scores": {k: round(v, 4) for k, v in r.scores.items()},
            "all_matches": r.all_matches,
            "matched_items": r.matched_items,
            "missed_items": r.missed_items,
            "excluded_matched": r.excluded_matched,
            "extra_matches": r.extra_matches
        }

    return {
        "results": [result_to_dict(r) for r in unique_results],
        "hints_used": hints_str
    }


# ============================================================================
# CLI
# ============================================================================

def print_pattern_result(result: dict, verbose: bool = True):
    """Print a single pattern result in a formatted way."""
    pattern = result["pattern"]
    source = result["source"].upper()
    total = result["total_score"]
    scores = result["scores"]

    # Status indicator
    is_perfect = (
        scores["matches_all"] == 1.0 and
        scores["excludes_all"] == 1.0
    )
    status = "OK" if is_perfect else "PARTIAL"

    print(f"\n  [{source}] {pattern}")
    print(f"  Status: {status} | Total Score: {total:.2%}")

    if verbose:
        # Individual scores
        print(f"  Scores: ", end="")
        score_parts = [f"{k}={v:.0%}" for k, v in scores.items()]
        print(", ".join(score_parts))

        # Match details
        if result["matched_items"]:
            print(f"  Matched: {result['matched_items']}")
        if result["missed_items"]:
            print(f"  Missed: {result['missed_items']}")
        if result["excluded_matched"]:
            print(f"  Bad (matched excludes): {result['excluded_matched']}")
        if result["extra_matches"]:
            print(f"  Extra: {result['extra_matches']}")


def run_tests(config: Config):
    """Run test suite with detailed output."""

    # Test cases designed for comparing LLM model behaviors
    # Each test targets a specific regex capability
    test_cases = [
        # ==========================================================
        # TIER 1: BASIC CAPABILITIES (models should pass all)
        # ==========================================================

        # T1.1: Character class - digits (\d)
        {
            "text": "Items: A1, B2, C3, D4, E5",
            "Highlighted Items": ["A1", "B2", "C3"],
            "Excluded Items": []
        },
        # T1.2: Character class - uppercase ([A-Z])
        {
            "text": "Mixed: HELLO world GOODBYE earth",
            "Highlighted Items": ["HELLO", "GOODBYE"],
            "Excluded Items": ["world", "earth"]
        },
        # T1.3: Simple separator pattern (word_digits)
        {
            "text": "IDs: user_01, admin_02, guest_03",
            "Highlighted Items": ["user_01", "admin_02", "guest_03"],
            "Excluded Items": []
        },

        # ==========================================================
        # TIER 2: WORD BOUNDARIES (key differentiator)
        # ==========================================================

        # T2.1: Substring exclusion - needs \b
        {
            "text": "test testing tested retest",
            "Highlighted Items": ["test"],
            "Excluded Items": ["testing", "tested", "retest"]
        },
        # T2.2: Prefix exclusion - needs boundary
        {
            "text": "ERR-01 ERR-02 ERR-001 ERR-002",
            "Highlighted Items": ["ERR-01", "ERR-02"],
            "Excluded Items": ["ERR-001", "ERR-002"]
        },
        # T2.3: Phone number boundary
        {
            "text": "Call 555-1234 or 555-5678, not 555-1234-5678",
            "Highlighted Items": ["555-1234", "555-5678"],
            "Excluded Items": ["555-1234-5678"]
        },

        # ==========================================================
        # TIER 3: QUANTIFIER PRECISION (moderate difficulty)
        # ==========================================================

        # T3.1: Fixed length digits
        {
            "text": "Codes: A01 A02 A03 A001 A002",
            "Highlighted Items": ["A01", "A02", "A03"],
            "Excluded Items": ["A001", "A002"]
        },
        # T3.2: Length range
        {
            "text": "Numbers: 12, 123, 1234, 12345",
            "Highlighted Items": ["123", "1234"],
            "Excluded Items": ["12", "12345"]
        },
        # T3.3: Version strings (multi-part)
        {
            "text": "Versions: v1.0, v2.1, v10.20, v1.0.0",
            "Highlighted Items": ["v1.0", "v2.1", "v10.20"],
            "Excluded Items": ["v1.0.0"]
        },

        # ==========================================================
        # TIER 4: COMPLEX EXCLUSIONS (harder)
        # ==========================================================

        # T4.1: Multiple exclusion criteria
        {
            "text": "Tags: #valid #good #123 #bad# #",
            "Highlighted Items": ["#valid", "#good", "#123"],
            "Excluded Items": ["#bad#", "#"]
        },
        # T4.2: Similar but different structures
        {
            "text": "Refs: item-001, thing-002, item_003, thing.004",
            "Highlighted Items": ["item-001", "thing-002"],
            "Excluded Items": ["item_003", "thing.004"]
        },
        # T4.3: Case + length combined
        {
            "text": "Keys: AB12, CD34, ab12, ABCD12",
            "Highlighted Items": ["AB12", "CD34"],
            "Excluded Items": ["ab12", "ABCD12"]
        },

        # ==========================================================
        # TIER 5: EDGE CASES (stress test)
        # ==========================================================

        # T5.1: Email-like patterns (structural distinction)
        {
            "text": "Contacts: a@b.co, x@y.io, test@, @bad.com",
            "Highlighted Items": ["a@b.co", "x@y.io"],
            "Excluded Items": ["test@", "@bad.com"]
        },
        # T5.2: Decimal precision (exactly 2 decimal places)
        {
            "text": "Prices: 10.99, 20.00, 5.5, 10.999",
            "Highlighted Items": ["10.99", "20.00"],
            "Excluded Items": ["5.5", "10.999"]
        },
    ]

    # Tier definitions (test indices)
    tiers = {
        "T1 Basic": [0, 1, 2],
        "T2 Word Boundaries": [3, 4, 5],
        "T3 Quantifiers": [6, 7, 8],
        "T4 Complex Exclusions": [9, 10, 11],
        "T5 Edge Cases": [12, 13],
    }

    print("Running test cases...\n")
    results = []

    for i, test_case in enumerate(test_cases):
        print(f"\n{'='*70}")
        print(f"TEST {i+1}: {test_case['Highlighted Items']} vs {test_case['Excluded Items']}")
        print(f"{'='*70}")
        print(f"Text: {test_case['text'][:60]}{'...' if len(test_case['text']) > 60 else ''}")

        try:
            result = generate_regex(test_case, config)

            # Find the LLM result specifically (not grex)
            llm_result = next((r for r in result["results"] if r["source"] == "llm"), None)
            is_pass = llm_result and llm_result["scores"]["matches_all"] == 1.0 and llm_result["scores"]["excludes_all"] == 1.0

            # Print each pattern result (LLM first, then grex for comparison)
            llm_results = [r for r in result["results"] if r["source"] == "llm"]
            grex_results = [r for r in result["results"] if r["source"] == "grex"]
            for pr in llm_results + grex_results:
                print_pattern_result(pr)

            results.append({
                "test": i + 1,
                "status": "pass" if is_pass else "partial",
                "result": result
            })

        except Exception as e:
            print(f"\n  FAILED: {e}")
            results.append({"test": i + 1, "status": "failed", "error": str(e)})

    # Summary with tier breakdown
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    passed = sum(1 for r in results if r["status"] == "pass")
    partial = sum(1 for r in results if r["status"] == "partial")
    failed = sum(1 for r in results if r["status"] == "failed")

    print(f"\nOverall: {passed}/{len(results)} passed, {partial} partial, {failed} failed")

    print(f"\nBy Tier:")
    for tier_name, indices in tiers.items():
        tier_passed = sum(1 for i in indices if i < len(results) and results[i]["status"] == "pass")
        print(f"  {tier_name}: {tier_passed}/{len(indices)}")

    # Detailed failures
    failures = [r for r in results if r["status"] != "pass"]
    if failures:
        print(f"\nFailed/Partial Tests:")
        for r in failures:
            test_idx = r["test"] - 1
            tc = test_cases[test_idx]
            print(f"  Test {r['test']}: {tc['Highlighted Items']} - {r['status']}")


def load_config_from_file(config_file: str, base_config: Config) -> Config:
    """Load config overrides from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            overrides = json.load(f)

        # Apply overrides to a copy of the config
        config = Config(
            model=overrides.get('model', base_config.model),
            temperature=overrides.get('temperature', base_config.temperature),
            max_attempts=overrides.get('max_attempts', base_config.max_attempts),
            reward_threshold=overrides.get('reward_threshold', base_config.reward_threshold),
            weights=overrides.get('weights', base_config.weights)
        )
        return config
    except Exception as e:
        print(f"[WARN] Failed to load config: {e}")
        return base_config


def list_dataset(output_file: str, config: Config):
    """Export the training dataset to a JSON file."""
    try:
        with open(config.dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Exported {len(data)} examples to {output_file}")
    except FileNotFoundError:
        # Write empty array if dataset doesn't exist
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"Dataset not found, exported empty array to {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to export dataset: {e}")
        sys.exit(1)


def add_example_to_dataset(example_file: str, config: Config):
    """Add a new example to the training dataset."""
    try:
        # Load the new example
        with open(example_file, 'r', encoding='utf-8') as f:
            new_example = json.load(f)

        # Load existing dataset
        try:
            with open(config.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []

        # Append the new example
        data.append(new_example)

        # Save updated dataset
        with open(config.dataset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Added example. Dataset now has {len(data)} examples.")
    except Exception as e:
        print(f"[ERROR] Failed to add example: {e}")
        sys.exit(1)


def delete_example_from_dataset(index: int, config: Config):
    """Delete an example from the training dataset by index."""
    try:
        # Load existing dataset
        with open(config.dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if index < 0 or index >= len(data):
            print(f"[ERROR] Index {index} out of range (0-{len(data)-1})")
            sys.exit(1)

        # Remove the example
        removed = data.pop(index)

        # Save updated dataset
        with open(config.dataset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print(f"Deleted example at index {index}. Dataset now has {len(data)} examples.")
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found: {config.dataset_file}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to delete example: {e}")
        sys.exit(1)


def main():
    """CLI entry point."""
    config = Config()

    # Parse arguments
    args = sys.argv[1:]

    # Handle --compile flag
    if len(args) == 1 and args[0] == "--compile":
        print("Pre-compiling regex generator...")
        print("This may take several minutes.\n")
        path = compile_and_save(config)
        if path:
            print(f"\nDone! Use the compiled program by running normally.")
        return

    # Handle --test flag
    if len(args) == 1 and args[0] == "--test":
        run_tests(config)
        return

    # Handle --list-dataset <output_file>
    if len(args) == 2 and args[0] == "--list-dataset":
        list_dataset(args[1], config)
        return

    # Handle --add-example <example_file>
    if len(args) == 2 and args[0] == "--add-example":
        add_example_to_dataset(args[1], config)
        return

    # Handle --delete-example <index>
    if len(args) == 2 and args[0] == "--delete-example":
        try:
            index = int(args[1])
        except ValueError:
            print(f"[ERROR] Index must be an integer, got: {args[1]}")
            sys.exit(1)
        delete_example_from_dataset(index, config)
        return

    # Handle input/output file mode with optional --config
    # Forms: <input> <output> OR <input> <output> --config <config_file>
    if len(args) >= 2 and not args[0].startswith("--"):
        input_file = args[0]
        output_file = args[1]

        # Check for --config flag
        if len(args) == 4 and args[2] == "--config":
            config = load_config_from_file(args[3], config)

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)

            if config.debug:
                print(f"Input: {json.dumps(input_data, indent=2)}")

            result = generate_regex(input_data, config)

        except Exception as e:
            result = {"error": str(e)}
            if config.debug:
                print(f"\n[ERROR] {e}")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

        if config.debug:
            print(f"\nOutput written to: {output_file}")
        return

    # Show usage
    print("Usage:")
    print("  python regespy.py --test                              # Run test cases")
    print("  python regespy.py --compile                           # Pre-compile for faster runtime")
    print("  python regespy.py <input.json> <output.json>          # Generate regex")
    print("  python regespy.py <input.json> <output.json> --config <config.json>  # With custom config")
    print("  python regespy.py --list-dataset <output.json>        # Export training dataset")
    print("  python regespy.py --add-example <example.json>        # Add example to dataset")
    print("  python regespy.py --delete-example <index>            # Delete example from dataset")
    sys.exit(1)


if __name__ == "__main__":
    main()