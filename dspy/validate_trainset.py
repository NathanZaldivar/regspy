#!/usr/bin/env python3
"""
Validate regex-dspy-train.json training set.

Checks:
1. Each pattern is valid regex syntax
2. Pattern matches all items in match_items
3. Pattern does NOT match any items in exclude_items
4. JSON structure is valid
"""

import json
import re
import sys
from pathlib import Path

def validate_training_set(filepath):
    """Validate the training set file."""

    print(f"Loading {filepath}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON: {e}")
        return False

    if not isinstance(data, list):
        print("ERROR: Training set must be a JSON array")
        return False

    print(f"Validating {len(data)} examples...\n")

    errors = []
    warnings = []
    passed = 0

    for idx, example in enumerate(data, 1):
        # Validate structure
        required_fields = ['text', 'match_items', 'exclude_items', 'expected_pattern']
        for field in required_fields:
            if field not in example:
                errors.append(f"Example {idx}: Missing field '{field}'")
                continue

        text = example.get('text', '')
        match_items = example.get('match_items', [])
        exclude_items = example.get('exclude_items', [])
        pattern_str = example.get('expected_pattern', '')

        # Try to compile the pattern
        try:
            pattern = re.compile(pattern_str)
        except re.error as e:
            errors.append(f"Example {idx}: Invalid regex pattern '{pattern_str}' - {e}")
            continue

        # Check that pattern matches all match_items
        for item in match_items:
            if not pattern.search(text):
                # Pattern doesn't match anything in text - check if item is findable
                if item not in text:
                    warnings.append(f"Example {idx}: Match item '{item}' not found in text")
                    continue

            if not re.search(pattern_str, item):
                errors.append(f"Example {idx}: Pattern '{pattern_str}' does NOT match required item '{item}'")
                continue

        # Check that pattern does NOT match any exclude_items
        for item in exclude_items:
            if re.search(pattern_str, item):
                errors.append(f"Example {idx}: Pattern '{pattern_str}' DOES match excluded item '{item}' (should not)")
                continue

        # If we got here, this example is valid
        passed += 1

    # Print results
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(data)} examples passed")
    print("=" * 70)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for error in errors:  # Show first 20 errors
            print(f"  [X] {error}")
    else:
        print("\n[OK] No errors found!")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10 warnings
            print(f"  [!] {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")

    print("\n" + "=" * 70)
    if errors:
        print("Validation FAILED - fix errors above")
        return False
    else:
        print("Validation PASSED!")
        return True

if __name__ == "__main__":
    filepath = Path(__file__).parent / "regex-dspy-train.json"
    success = validate_training_set(str(filepath))
    sys.exit(0 if success else 1)
