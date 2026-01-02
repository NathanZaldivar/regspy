/* ===========================================
   Regex Generator JavaScript

   Handles:
   - Tab navigation (Generate, Results, Dataset, Config)
   - Text selection/highlighting (positive examples - cyan)
   - Right-click to toggle exclude mode (negative examples - red)
   - Multiple selection management with match/exclude states
   - Clipboard content display
   - Results table with sorting and medals
   - Dataset management
   - Session config editing
   - AHK backend communication for LLM regex generation
   - Loading states and error display
   =========================================== */

// ===========================================
// STATE
// ===========================================

/** @type {{text: string, start: number, end: number, negative: boolean}[]} Array of user-selected text snippets with position and match/exclude flag */
let selections = [];

/** @type {string} Original clipboard text */
let originalText = '';

/** @type {boolean} Loading state flag */
let isLoading = false;

/** @type {string} Current active tab */
let currentTab = 'generate';

/** @type {Object|null} Current results data from last generation */
let currentResults = null;

/** @type {Object|null} Current dataset loaded from file */
let currentDataset = null;

/** @type {string} Current sort column for results table */
let resultsSortColumn = 'total_score';

/** @type {string} Current sort direction ('asc' or 'desc') */
let resultsSortDirection = 'desc';

/** @type {string} Current sort column for dataset table */
let datasetSortColumn = 'index';

/** @type {string} Current sort direction for dataset ('asc' or 'desc') */
let datasetSortDirection = 'asc';

/** @type {Set<number>} Set of result indices that have been added to dataset */
let addedToDataset = new Set();

/** @type {Set<string>} Set of expanded row IDs (e.g., 'result-0', 'dataset-5') */
let expandedRows = new Set();

/** @type {Object} Default config values */
const defaultConfig = {
    model: 'qwen2.5:3b',
    temperature: 0.4,
    max_attempts: 10,
    reward_threshold: 0.85,
    weights: {
        matches_all: 0.35,
        excludes_all: 0.25,
        coherence: 0.15,
        generalization: 0.15,
        simplicity: 0.10
    }
};


// ===========================================
// DOM ELEMENTS
// ===========================================

const textbox = document.getElementById('textbox');
const selectionList = document.getElementById('selectionList');
const matchCountDisplay = document.getElementById('matchCount');
const excludeCountDisplay = document.getElementById('excludeCount');
const submitBtn = document.getElementById('submitBtn');
const clearBtn = document.getElementById('clearBtn');
const cancelBtn = document.getElementById('cancelBtn');

// Tab elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Results tab elements
const resultsEmptyState = document.querySelector('.results-empty-state');
const resultsTableContainer = document.getElementById('resultsTableContainer');

// Dataset tab elements
const datasetCount = document.getElementById('datasetCount');
const datasetSearch = document.getElementById('datasetSearch');
const datasetTableContainer = document.getElementById('datasetTableContainer');
const refreshDatasetBtn = document.getElementById('refreshDataset');

// Config tab elements
const configModel = document.getElementById('configModel');
const configTemperature = document.getElementById('configTemperature');
const configMaxAttempts = document.getElementById('configMaxAttempts');
const configRewardThreshold = document.getElementById('configRewardThreshold');
const weightInputs = {
    matches_all: document.getElementById('weightMatchesAll'),
    excludes_all: document.getElementById('weightExcludesAll'),
    coherence: document.getElementById('weightCoherence'),
    generalization: document.getElementById('weightGeneralization'),
    simplicity: document.getElementById('weightSimplicity')
};
const weightTotal = document.getElementById('weightTotal');
const resetConfigBtn = document.getElementById('resetConfig');

// Modal elements
const modalOverlay = document.getElementById('modalOverlay');
const modalTitle = document.getElementById('modalTitle');
const modalContent = document.getElementById('modalContent');
const modalCancel = document.getElementById('modalCancel');
const modalConfirm = document.getElementById('modalConfirm');
const modalClose = document.querySelector('.modal-close');

/** @type {Function|null} Current modal confirm callback */
let modalConfirmCallback = null;

// ===========================================
// AHK -> JS FUNCTIONS
// These are called from AutoHotkey via ExecuteScriptAsync
// ===========================================

/**
 * Handle textbox input changes.
 * Tracks the original text and clears selections when content changes.
 */
function handleTextboxInput() {
    // Normalize newlines to \n (Windows uses \r\n which throws off positions)
    const newText = (textbox.textContent || '').replace(/\r\n/g, '\n').replace(/\r/g, '\n');

    // If text changed, clear selections (they're no longer valid)
    if (newText !== originalText) {
        originalText = newText;
        selections = [];
        updateUI();
        // Re-render with normalized text
        textbox.textContent = originalText;
    }
}

// Listen for input changes on the editable textbox
textbox.addEventListener('input', handleTextboxInput);

// Handle paste events - strip HTML formatting, keep only plain text
textbox.addEventListener('paste', function(e) {
    e.preventDefault();

    // Get plain text from clipboard (strips all HTML formatting)
    const text = (e.clipboardData || window.clipboardData).getData('text/plain');

    // Insert plain text at cursor position
    const selection = window.getSelection();
    if (!selection.rangeCount) return;

    const range = selection.getRangeAt(0);
    range.deleteContents();
    range.insertNode(document.createTextNode(text));

    // Move cursor to end of inserted text
    selection.collapseToEnd();

    handleTextboxInput();
});

/**
 * Handle regex generation result from AHK.
 * Called when regex patterns are successfully generated.
 * Appends new results to existing ones (preserves history).
 *
 * @param {string} resultsJson - JSON string with results array and hints
 */
window.onRegexGenerated = function(resultsJson) {
    try {
        const data = JSON.parse(resultsJson);

        if (data && data.results && data.results.length > 0) {
            // Mark new results with a run timestamp
            const runTime = new Date().toLocaleTimeString();
            data.results.forEach(r => r._runTime = runTime);

            if (currentResults && currentResults.results) {
                // Append new results to existing (new ones first)
                currentResults.results = [...data.results, ...currentResults.results];
                currentResults.hints_used = data.hints_used; // Update hints to latest
            } else {
                // First run - just store
                currentResults = data;
            }

            // Render results table
            renderResultsTable();

            // Switch to results tab
            switchTab('results');
        } else {
            showErrorInUi('No patterns were generated');
        }
    } catch (e) {
        console.error('Failed to parse results:', e);
        showErrorInUi('Failed to parse generated results');
    }
};

/**
 * Handle dataset loaded from AHK.
 * Called when dataset is fetched from Python.
 *
 * @param {string} datasetJson - JSON string array of training examples
 */
window.onDatasetLoaded = function(datasetJson) {
    try {
        currentDataset = JSON.parse(datasetJson);
        renderDatasetTable();
    } catch (e) {
        console.error('Failed to parse dataset:', e);
        datasetTableContainer.innerHTML = '<div class="dataset-loading">Failed to load dataset</div>';
    }
};

/**
 * Confirm that an example was added to the dataset.
 * Called from AHK after successful addition.
 *
 * @param {number} resultIndex - Index of the result that was added
 */
window.onExampleAdded = function(resultIndex) {
    addedToDataset.add(resultIndex);
    renderResultsTable();
};

/**
 * Confirm that a dataset item was deleted.
 * Called from AHK after successful deletion.
 *
 * @param {number} index - Index that was deleted
 */
window.onDatasetItemDeleted = function(index) {
    // Refresh the dataset
    if (typeof ahk !== 'undefined' && ahk.loadDataset) {
        ahk.loadDataset();
    }
};

/**
 * Set loading state for the UI.
 * Called from AHK to show/hide loading indicator.
 * 
 * @param {boolean} loading - Whether to show loading state
 */
window.setLoading = function(loading) {
    isLoading = loading;
    updateLoadingUI();
};

/**
 * Display an error message to the user.
 * Called from AHK when an error occurs.
 * 
 * @param {string} message - Error message to display
 */
window.showError = function(message) {
    showErrorInUi(message);
};

// ===========================================
// SELECTION HANDLING
// ===========================================

/**
 * Calculate the absolute text offset within a container element.
 * Uses the container's textContent to find the position.
 *
 * @param {Node} container - The container element
 * @param {Node} targetNode - The node where the offset is
 * @param {number} offset - The offset within targetNode
 * @returns {number} Absolute character offset from start of container's text
 */
function getTextOffset(container, targetNode, offset) {
    // Method 1: Create a range from start to target and measure its text
    try {
        const range = document.createRange();
        range.setStart(container, 0);
        range.setEnd(targetNode, offset);
        return range.toString().length;
    } catch (e) {
        // Method 2: Fallback - walk the DOM manually
        let total = 0;
        let found = false;

        function walk(node) {
            if (found) return;

            if (node.nodeType === Node.TEXT_NODE) {
                if (node === targetNode) {
                    total += offset;
                    found = true;
                    return;
                }
                total += node.textContent.length;
            } else {
                for (const child of node.childNodes) {
                    walk(child);
                    if (found) return;
                }
            }
        }

        walk(container);
        return total;
    }
}

/**
 * Handle text selection in the textbox.
 * Captures selected text with position and adds to selections array.
 * Prevents selection during loading state.
 */
textbox.addEventListener('mouseup', function() {
    // Don't allow selections while loading
    if (isLoading) return;

    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    // Ignore empty selections
    if (!selectedText) return;

    // Get the range to calculate approximate position
    if (!selection.rangeCount) return;
    const range = selection.getRangeAt(0);

    // Strategy: Count how many occurrences START before the selection point
    // This handles overlapping occurrences correctly

    // Get the approximate position using textBefore length
    const beforeRange = document.createRange();
    beforeRange.setStart(textbox, 0);
    beforeRange.setEnd(range.startContainer, range.startOffset);
    const approxStart = beforeRange.toString().length;

    // Find ALL occurrences of selectedText in originalText
    const occurrences = [];
    let searchPos = 0;
    while (true) {
        const idx = originalText.indexOf(selectedText, searchPos);
        if (idx === -1) break;
        occurrences.push({ start: idx, end: idx + selectedText.length });
        searchPos = idx + 1;
    }

    if (occurrences.length === 0) {
        selection.removeAllRanges();
        return;
    }

    // Count occurrences that START before approxStart
    let countBefore = 0;
    for (const occ of occurrences) {
        if (occ.start < approxStart) {
            countBefore++;
        }
    }

    // The selected occurrence is at index countBefore (0-indexed)
    let targetOccurrence = null;

    if (countBefore < occurrences.length) {
        targetOccurrence = occurrences[countBefore];
    }

    // If that occurrence is already selected, or countBefore was off, find closest available
    if (!targetOccurrence || selections.some(s => s.start === targetOccurrence.start && s.end === targetOccurrence.end)) {
        // Fallback: use approximate position
        const approxStart = getTextOffset(textbox, range.startContainer, range.startOffset);

        const available = occurrences.filter(occ =>
            !selections.some(s => s.start === occ.start && s.end === occ.end)
        );

        if (available.length === 0) {
            selection.removeAllRanges();
            return;
        }

        let bestMatch = available[0];
        let bestDistance = Math.abs(available[0].start - approxStart);

        for (const occ of available) {
            const distance = Math.abs(occ.start - approxStart);
            if (distance < bestDistance) {
                bestDistance = distance;
                bestMatch = occ;
            }
        }

        targetOccurrence = bestMatch;
    }

    const start = targetOccurrence.start;
    const end = targetOccurrence.end;

    // Add to selections with position info
    addSelection(selectedText, start, end);

    // Clear browser selection highlight
    selection.removeAllRanges();
});

/**
 * Add a selection to the list.
 *
 * @param {string} text - Selected text to add
 * @param {number} start - Start position in original text
 * @param {number} end - End position in original text
 */
function addSelection(text, start, end) {
    selections.push({ text, start, end, negative: false });
    updateUI();
    highlightSelections();
}

/**
 * Toggle a selection between positive (match) and negative (exclude).
 *
 * @param {number} index - Index of selection to toggle
 */
function toggleNegative(index) {
    if (index >= 0 && index < selections.length) {
        selections[index].negative = !selections[index].negative;
        updateUI();
        highlightSelections();
    }
}

/**
 * Remove a selection by index.
 * 
 * @param {number} index - Index of selection to remove
 */
function removeSelection(index) {
    selections.splice(index, 1);
    updateUI();
    highlightSelections();
}

/**
 * Clear all selections.
 */
function clearSelections() {
    selections = [];
    updateUI();
    highlightSelections();
}

// ===========================================
// UI UPDATES
// ===========================================

/**
 * Update all UI elements to reflect current state.
 */
function updateUI() {
    // Count positive and negative selections
    const positiveCount = selections.filter(s => !s.negative).length;
    const negativeCount = selections.filter(s => s.negative).length;

    // Update selection count displays
    matchCountDisplay.textContent = positiveCount;
    excludeCountDisplay.textContent = negativeCount;

    // Update submit button state (disabled if no positive selections or loading)
    submitBtn.disabled = positiveCount === 0 || isLoading;

    // Update selection list panel
    renderSelectionList();
}

/**
 * Update UI elements for loading state.
 */
function updateLoadingUI() {
    const positiveCount = selections.filter(s => !s.negative).length;

    // Update button text and state
    if (isLoading) {
        submitBtn.textContent = 'Generating...';
        submitBtn.disabled = true;
        textbox.style.opacity = '0.6';
        textbox.style.pointerEvents = 'none';
    } else {
        submitBtn.textContent = 'Generate Regex';
        submitBtn.disabled = positiveCount === 0;
        textbox.style.opacity = '1';
        textbox.style.pointerEvents = 'auto';
    }
}

/**
 * Render the list of current selections.
 */
function renderSelectionList() {
    selectionList.innerHTML = '';

    selections.forEach((sel, index) => {
        const li = document.createElement('li');
        li.className = 'selection-item' + (sel.negative ? ' negative' : '');
        li.dataset.index = index;

        // Truncate long selections for display
        const displayText = sel.text.length > 40
            ? sel.text.substring(0, 40) + '...'
            : sel.text;

        const label = sel.negative ? 'exclude' : 'match';

        li.innerHTML = `
            <span class="selection-label">${label}</span>
            <span class="selection-text" title="${escapeHtml(sel.text)}">${escapeHtml(displayText)}</span>
            <button class="selection-remove" data-index="${index}">&times;</button>
        `;

        selectionList.appendChild(li);
    });
}

/**
 * Highlight all selections in the textbox.
 * Uses position-based highlighting for exact placement.
 */
function highlightSelections() {
    if (selections.length === 0) {
        // No selections - show plain text (CSS white-space: pre-wrap preserves newlines)
        textbox.textContent = originalText;
        return;
    }

    // Sort selections by start position
    const sorted = selections
        .map((sel, index) => ({ ...sel, index }))
        .sort((a, b) => a.start - b.start);

    // Build HTML by inserting spans at exact positions
    let html = '';
    let lastEnd = 0;

    for (const sel of sorted) {
        // Skip if this selection overlaps with a previous one
        if (sel.start < lastEnd) continue;

        // Add text before this selection
        if (sel.start > lastEnd) {
            html += escapeHtml(originalText.substring(lastEnd, sel.start));
        }

        // Add the highlighted selection - use substring from original to ensure exact match
        const className = sel.negative ? 'highlight highlight-negative' : 'highlight';
        const spanText = originalText.substring(sel.start, sel.end);
        html += `<span class="${className}" data-index="${sel.index}">${escapeHtml(spanText)}</span>`;

        lastEnd = sel.end;
    }

    // Add remaining text after last selection
    if (lastEnd < originalText.length) {
        html += escapeHtml(originalText.substring(lastEnd));
    }

    textbox.innerHTML = html;

    // Keep originalText in sync with DOM (in case browser normalized anything)
    originalText = textbox.textContent;
}


/**
 * Display an error message to the user.
 * 
 * @param {string} message - Error message to display
 */
function showErrorInUi(message) {
    // Remove existing error if any
    hideError();
    
    // Create error element
    const errorDiv = document.createElement('div');
    errorDiv.id = 'errorMessage';
    errorDiv.className = 'error-message';
    errorDiv.innerHTML = `
        <span class="error-icon">⚠️</span>
        <span class="error-text">${escapeHtml(message)}</span>
        <button class="error-close">&times;</button>
    `;
    
    // Insert at top of content area
    const content = document.querySelector('.content');
    content.insertBefore(errorDiv, content.firstChild);
    
    // Auto-hide after 5 seconds
    setTimeout(hideError, 5000);
    
    // Close button handler
    errorDiv.querySelector('.error-close').addEventListener('click', hideError);
}

/**
 * Hide/remove the error message.
 */
function hideError() {
    const existing = document.getElementById('errorMessage');
    if (existing) {
        existing.remove();
    }
}

/**
 * Copy text to clipboard using modern API or fallback.
 * 
 * @param {string} text - Text to copy
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
    } catch (e) {
        // Fallback for older browsers
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
}

// ===========================================
// EVENT HANDLERS
// ===========================================

// Clear all selections button
clearBtn.addEventListener('click', clearSelections);

// Cancel button - close window without action
cancelBtn.addEventListener('click', function() {
    if (typeof ahk !== 'undefined' && ahk.cancel) {
        ahk.cancel();
    }
});

// Submit button - send selections to AHK for regex generation
submitBtn.addEventListener('click', function() {
    const positiveSelections = selections.filter(s => !s.negative).map(s => s.text);
    const negativeSelections = selections.filter(s => s.negative).map(s => s.text);

    // Prevent action if no positive selections or already loading
    if (positiveSelections.length === 0 || isLoading) return;

    // Hide any previous errors
    hideError();

    // Get session config
    const config = getSessionConfig();

    // Send original text + selections + config to AHK
    // AHK will call the Python script and handle the result
    if (typeof ahk !== 'undefined' && ahk.generateRegex) {
        ahk.generateRegex(
            originalText,
            JSON.stringify(positiveSelections),
            JSON.stringify(negativeSelections),
            JSON.stringify(config)
        );
    } else {
        // Development fallback - log what would be sent
        console.log('AHK not available. Would send:', {
            text: originalText,
            positiveSelections,
            negativeSelections,
            config
        });
    }
});

// Handle clicks on remove buttons in selection list (event delegation)
selectionList.addEventListener('click', function(e) {
    if (e.target.classList.contains('selection-remove')) {
        const index = parseInt(e.target.dataset.index, 10);
        removeSelection(index);
    }
});

// Handle right-click on selection list items to toggle negative
selectionList.addEventListener('contextmenu', function(e) {
    const item = e.target.closest('.selection-item');
    if (item) {
        e.preventDefault();
        const index = parseInt(item.dataset.index, 10);
        toggleNegative(index);
    }
});

// Handle clicks on highlights to remove them
textbox.addEventListener('click', function(e) {
    if (e.target.classList.contains('highlight')) {
        const index = parseInt(e.target.dataset.index, 10);
        if (index !== -1) {
            removeSelection(index);
        }
    }
});

// Handle right-click on highlights to toggle negative
textbox.addEventListener('contextmenu', function(e) {
    if (e.target.classList.contains('highlight')) {
        e.preventDefault();
        const index = parseInt(e.target.dataset.index, 10);
        if (index !== -1) {
            toggleNegative(index);
        }
    }
});

// ===========================================
// UTILITY FUNCTIONS
// ===========================================

/**
 * Escape HTML special characters to prevent XSS.
 * 
 * @param {string} text - Text to escape
 * @returns {string} HTML-escaped text
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Escape special regex metacharacters.
 * 
 * @param {string} text - Text to escape
 * @returns {string} Regex-safe text
 */
function escapeRegex(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// ===========================================
// TAB NAVIGATION
// ===========================================

/**
 * Switch to a different tab.
 * @param {string} tabId - The tab to switch to ('generate', 'results', 'dataset', 'config')
 */
function switchTab(tabId) {
    currentTab = tabId;

    // Update tab buttons
    tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    // Update tab contents
    tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `tab-${tabId}`);
    });

    // Update footer buttons based on tab
    updateFooterForTab(tabId);

    // Trigger tab-specific actions
    if (tabId === 'dataset' && !currentDataset) {
        // Load dataset when first visiting the tab
        if (typeof ahk !== 'undefined' && ahk.loadDataset) {
            ahk.loadDataset();
        }
    }
}

/**
 * Update footer buttons based on current tab.
 * @param {string} tabId - Current tab ID
 */
function updateFooterForTab(tabId) {
    const positiveCount = selections.filter(s => !s.negative).length;

    if (tabId === 'generate') {
        submitBtn.textContent = 'Generate Regex';
        submitBtn.disabled = positiveCount === 0 || isLoading;
        submitBtn.style.display = '';
    } else {
        submitBtn.style.display = 'none';
    }
}

// Tab button click handlers
tabButtons.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ===========================================
// RESULTS TABLE
// ===========================================

/**
 * Render the results table with current data.
 */
function renderResultsTable() {
    const resultsHeader = document.querySelector('.results-header');
    const resultsCount = document.getElementById('resultsCount');

    if (!currentResults || !currentResults.results.length) {
        resultsEmptyState.style.display = '';
        resultsTableContainer.style.display = 'none';
        if (resultsHeader) resultsHeader.style.display = 'none';
        return;
    }

    resultsEmptyState.style.display = 'none';
    resultsTableContainer.style.display = '';
    if (resultsHeader) resultsHeader.style.display = '';
    if (resultsCount) resultsCount.textContent = currentResults.results.length;

    // Sort results
    const sortedResults = [...currentResults.results].sort((a, b) => {
        let aVal = getSortValue(a, resultsSortColumn);
        let bVal = getSortValue(b, resultsSortColumn);

        if (resultsSortDirection === 'asc') {
            return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        } else {
            return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        }
    });

    // Build table HTML
    const headers = [
        { key: 'rank', label: 'Rank', sortable: false },
        { key: 'pattern', label: 'Pattern', sortable: true },
        { key: 'total_score', label: 'Score', sortable: true },
        { key: 'matches', label: 'Matches', sortable: true },
        { key: 'missed', label: 'Missed', sortable: true },
        { key: 'excluded', label: 'Excluded', sortable: true },
        { key: 'source', label: 'Source', sortable: true },
        { key: 'actions', label: 'Actions', sortable: false }
    ];

    const headerHtml = headers.map(h => {
        const sortClass = h.key === resultsSortColumn ? 'sorted' : '';
        const sortIndicator = h.sortable ?
            `<span class="sort-indicator">${resultsSortColumn === h.key ? (resultsSortDirection === 'asc' ? '▲' : '▼') : '⇅'}</span>` : '';
        const clickAttr = h.sortable ? `data-sort="${h.key}"` : '';
        return `<th class="${sortClass}" ${clickAttr}>${h.label}${sortIndicator}</th>`;
    }).join('');

    const rowsHtml = sortedResults.map((result, sortedIndex) => {
        // Find original index for add-to-dataset tracking
        const originalIndex = currentResults.results.indexOf(result);
        const isAdded = addedToDataset.has(originalIndex);
        const rowId = `result-${originalIndex}`;
        const isExpanded = expandedRows.has(rowId);

        // Medal based on sorted position (by score)
        const medal = getMedalIcon(sortedIndex);

        // Pattern (truncate if too long)
        const patternDisplay = result.pattern.length > 30
            ? result.pattern.substring(0, 30) + '...'
            : result.pattern;

        // Score display
        const scorePercent = (result.total_score * 100).toFixed(1);
        const scoreClass = result.total_score >= 0.9 ? '' :
            result.total_score >= 0.7 ? 'partial' : 'fail';

        // Counts
        const matchCount = result.matched_items?.length || 0;
        const totalMatch = (result.matched_items?.length || 0) + (result.missed_items?.length || 0);
        const missedCount = result.missed_items?.length || 0;
        const excludedCount = result.excluded_matched?.length || 0;

        // Source badge
        const sourceClass = result.source === 'llm' ? 'llm' : 'grex';

        // Star button state
        const starBtn = isAdded
            ? `<button class="action-btn added" disabled title="Added to dataset">✓</button>`
            : `<button class="action-btn star-btn" data-index="${originalIndex}" title="Add to dataset">⭐</button>`;

        // Expand indicator
        const expandIcon = isExpanded ? '▼' : '▶';

        // Human-readable score labels with tooltips
        const scoreLabels = {
            matches_all: { label: 'Matches All', title: 'Percentage of required items the pattern matches' },
            excludes_all: { label: 'Excludes All', title: 'Percentage of excluded items the pattern avoids' },
            coherence: { label: 'Coherence', title: 'How similar extra matches are to target items' },
            generalization: { label: 'Generalization', title: 'Use of character classes (\\d, \\w) vs literals' },
            simplicity: { label: 'Simplicity', title: 'Shorter patterns with less branching score higher' }
        };

        // Build detail row content with improved organization
        const detailHtml = isExpanded ? `
            <tr class="detail-row" data-parent="${rowId}">
                <td colspan="8">
                    <div class="detail-content">
                        <!-- Pattern Section -->
                        <div class="detail-card">
                            <div class="detail-card-header">Pattern</div>
                            <div class="detail-value pattern">${escapeHtml(result.pattern)}</div>
                        </div>

                        <div class="detail-columns">
                            <!-- Score Breakdown Section -->
                            <div class="detail-card">
                                <div class="detail-card-header">Score Breakdown</div>
                                <div class="score-breakdown">
                                    ${Object.entries(result.scores || {}).map(([k, v]) => {
                                        const info = scoreLabels[k] || { label: k, title: k };
                                        const pct = (v * 100).toFixed(0);
                                        return `
                                            <div class="score-row" title="${info.title}">
                                                <span class="score-label">${info.label}</span>
                                                <span class="score-pct">${pct}%</span>
                                                <div class="score-bar"><div class="score-fill" style="width: ${pct}%"></div></div>
                                            </div>
                                        `;
                                    }).join('')}
                                </div>
                            </div>

                            <!-- Match Results Section -->
                            <div class="detail-card">
                                <div class="detail-card-header">Match Results</div>
                                <div class="match-results">
                                    <div class="match-row">
                                        <span class="match-icon success">✓</span>
                                        <span class="match-label">Matched:</span>
                                        <span class="match-value">${result.matched_items?.join(', ') || '(none)'}</span>
                                    </div>
                                    <div class="match-row ${result.missed_items?.length ? 'has-error' : ''}">
                                        <span class="match-icon error">✗</span>
                                        <span class="match-label">Missed:</span>
                                        <span class="match-value">${result.missed_items?.join(', ') || '(none)'}</span>
                                    </div>
                                    <div class="match-row ${result.excluded_matched?.length ? 'has-error' : ''}">
                                        <span class="match-icon warning">⚠</span>
                                        <span class="match-label">Bad Excludes:</span>
                                        <span class="match-value">${result.excluded_matched?.join(', ') || '(none)'}</span>
                                    </div>
                                    <div class="match-row">
                                        <span class="match-icon neutral">○</span>
                                        <span class="match-label">Extra:</span>
                                        <span class="match-value">${result.extra_matches?.join(', ') || '(none)'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        ` : '';

        return `
            <tr class="expandable-row ${isExpanded ? 'expanded' : ''}" data-row-id="${rowId}" data-index="${originalIndex}">
                <td><span class="expand-icon">${expandIcon}</span><span class="rank-medal ${medal.class}">${medal.icon}</span></td>
                <td class="pattern-cell" title="${escapeHtml(result.pattern)}">${escapeHtml(patternDisplay)}</td>
                <td class="score-cell ${scoreClass}">${scorePercent}%</td>
                <td>${matchCount}/${totalMatch}</td>
                <td class="${missedCount > 0 ? 'score-cell fail' : ''}">${missedCount}</td>
                <td class="${excludedCount > 0 ? 'score-cell fail' : ''}">${excludedCount}</td>
                <td><span class="source-badge ${sourceClass}">${result.source}</span></td>
                <td>
                    ${starBtn}
                    <button class="action-btn copy-btn" data-pattern="${escapeHtml(result.pattern)}" title="Copy pattern">📋</button>
                </td>
            </tr>
            ${detailHtml}
        `;
    }).join('');

    resultsTableContainer.innerHTML = `
        <table class="results-table">
            <thead><tr>${headerHtml}</tr></thead>
            <tbody>${rowsHtml}</tbody>
        </table>
    `;

    // Add event listeners
    resultsTableContainer.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (resultsSortColumn === col) {
                resultsSortDirection = resultsSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                resultsSortColumn = col;
                resultsSortDirection = 'desc';
            }
            renderResultsTable();
        });
    });

    resultsTableContainer.querySelectorAll('.star-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Don't trigger row expansion
            const index = parseInt(btn.dataset.index, 10);
            showAddToDatasetModal(index);
        });
    });

    resultsTableContainer.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Don't trigger row expansion
            copyToClipboard(btn.dataset.pattern);
            btn.textContent = '✓';
            setTimeout(() => { btn.textContent = '📋'; }, 1000);
        });
    });

    // Expandable row click handlers
    resultsTableContainer.querySelectorAll('.expandable-row').forEach(row => {
        row.addEventListener('click', (e) => {
            // Don't expand if clicking on action buttons
            if (e.target.closest('.action-btn')) return;

            const rowId = row.dataset.rowId;
            if (expandedRows.has(rowId)) {
                expandedRows.delete(rowId);
            } else {
                expandedRows.add(rowId);
            }
            renderResultsTable();
        });
    });
}

/**
 * Get sort value for a result by column.
 */
function getSortValue(result, column) {
    switch (column) {
        case 'pattern': return result.pattern.toLowerCase();
        case 'total_score': return result.total_score;
        case 'matches': return result.matched_items?.length || 0;
        case 'missed': return result.missed_items?.length || 0;
        case 'excluded': return result.excluded_matched?.length || 0;
        case 'source': return result.source;
        default: return 0;
    }
}

/**
 * Get medal icon for rank.
 */
function getMedalIcon(index) {
    switch (index) {
        case 0: return { icon: '🥇', class: 'gold' };
        case 1: return { icon: '🥈', class: 'silver' };
        case 2: return { icon: '🥉', class: 'bronze' };
        default: return { icon: `#${index + 1}`, class: 'none' };
    }
}

// ===========================================
// ADD TO DATASET MODAL
// ===========================================

/**
 * Show the add to dataset confirmation modal.
 * @param {number} resultIndex - Index of the result to add
 */
function showAddToDatasetModal(resultIndex) {
    const result = currentResults.results[resultIndex];

    modalTitle.textContent = 'Add to Training Dataset?';

    // Build preview content
    const matchItems = selections.filter(s => !s.negative).map(s => s.text);
    const excludeItems = selections.filter(s => s.negative).map(s => s.text);

    modalContent.innerHTML = `
        <div class="preview-item">
            <div class="preview-label">Text</div>
            <div class="preview-value">${escapeHtml(truncateText(originalText, 100))}</div>
        </div>
        <div class="preview-item">
            <div class="preview-label">Match Items</div>
            <div class="preview-value">${matchItems.length ? escapeHtml(matchItems.join(', ')) : '(none)'}</div>
        </div>
        <div class="preview-item">
            <div class="preview-label">Exclude Items</div>
            <div class="preview-value">${excludeItems.length ? escapeHtml(excludeItems.join(', ')) : '(none)'}</div>
        </div>
        <div class="preview-item">
            <div class="preview-label">Pattern</div>
            <div class="preview-value pattern">${escapeHtml(result.pattern)}</div>
        </div>
    `;

    modalConfirm.textContent = 'Add';

    modalConfirmCallback = () => {
        const example = {
            text: originalText,
            match_items: matchItems,
            exclude_items: excludeItems,
            expected_pattern: result.pattern
        };

        if (typeof ahk !== 'undefined' && ahk.addToDataset) {
            ahk.addToDataset(JSON.stringify(example), resultIndex);
        }
    };

    showModal();
}

// ===========================================
// DATASET TABLE
// ===========================================

/**
 * Get sort value for a dataset item by column.
 */
function getDatasetSortValue(example, originalIndex, column) {
    switch (column) {
        case 'index': return originalIndex;
        case 'text': return (example.text || '').toLowerCase();
        case 'match_count': return example.match_items?.length || 0;
        case 'exclude_count': return example.exclude_items?.length || 0;
        case 'pattern': return (example.expected_pattern || '').toLowerCase();
        default: return 0;
    }
}

/**
 * Render the dataset table with current data.
 */
function renderDatasetTable() {
    if (!currentDataset || !currentDataset.length) {
        datasetTableContainer.innerHTML = '<div class="dataset-loading">No training examples found</div>';
        datasetCount.textContent = '0';
        return;
    }

    // Filter by search
    const searchTerm = datasetSearch.value.toLowerCase();
    const filteredData = searchTerm
        ? currentDataset.map((ex, i) => ({ example: ex, originalIndex: i }))
            .filter(({ example }) =>
                example.text?.toLowerCase().includes(searchTerm) ||
                example.expected_pattern?.toLowerCase().includes(searchTerm) ||
                example.match_items?.some(m => m.toLowerCase().includes(searchTerm))
            )
        : currentDataset.map((ex, i) => ({ example: ex, originalIndex: i }));

    // Sort
    const sortedData = [...filteredData].sort((a, b) => {
        const aVal = getDatasetSortValue(a.example, a.originalIndex, datasetSortColumn);
        const bVal = getDatasetSortValue(b.example, b.originalIndex, datasetSortColumn);

        if (datasetSortDirection === 'asc') {
            return aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
        } else {
            return aVal < bVal ? 1 : aVal > bVal ? -1 : 0;
        }
    });

    datasetCount.textContent = currentDataset.length;

    // Build headers with sort indicators
    const headers = [
        { key: 'index', label: '#', sortable: true },
        { key: 'text', label: 'Text', sortable: true },
        { key: 'match_count', label: 'Match Items', sortable: true },
        { key: 'exclude_count', label: 'Exclude Items', sortable: true },
        { key: 'pattern', label: 'Pattern', sortable: true },
        { key: 'actions', label: 'Actions', sortable: false }
    ];

    const headerHtml = headers.map(h => {
        const sortClass = h.key === datasetSortColumn ? 'sorted' : '';
        const sortIndicator = h.sortable ?
            `<span class="sort-indicator">${datasetSortColumn === h.key ? (datasetSortDirection === 'asc' ? '▲' : '▼') : '⇅'}</span>` : '';
        const clickAttr = h.sortable ? `data-sort="${h.key}"` : '';
        return `<th class="${sortClass}" ${clickAttr}>${h.label}${sortIndicator}</th>`;
    }).join('');

    const rowsHtml = sortedData.map(({ example, originalIndex }) => {
        const rowId = `dataset-${originalIndex}`;
        const isExpanded = expandedRows.has(rowId);
        const textDisplay = truncateText(example.text || '', 50);
        const matchCount = example.match_items?.length || 0;
        const excludeCount = example.exclude_items?.length || 0;
        const patternDisplay = truncateText(example.expected_pattern || '', 30);

        const expandIcon = isExpanded ? '▼' : '▶';

        // Build detail row content
        const detailHtml = isExpanded ? `
            <tr class="detail-row" data-parent="${rowId}">
                <td colspan="6">
                    <div class="detail-content">
                        <div class="detail-section">
                            <div class="detail-label">Full Text</div>
                            <div class="detail-value">${escapeHtml(example.text || '')}</div>
                        </div>
                        <div class="detail-grid">
                            <div class="detail-section">
                                <div class="detail-label">Match Items (${matchCount})</div>
                                <div class="detail-value">${(example.match_items || []).map(m => escapeHtml(m)).join(', ') || '(none)'}</div>
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Exclude Items (${excludeCount})</div>
                                <div class="detail-value">${(example.exclude_items || []).map(m => escapeHtml(m)).join(', ') || '(none)'}</div>
                            </div>
                            <div class="detail-section">
                                <div class="detail-label">Expected Pattern</div>
                                <div class="detail-value pattern">${escapeHtml(example.expected_pattern || '')}</div>
                            </div>
                        </div>
                    </div>
                </td>
            </tr>
        ` : '';

        return `
            <tr class="expandable-row ${isExpanded ? 'expanded' : ''}" data-row-id="${rowId}" data-index="${originalIndex}">
                <td><span class="expand-icon">${expandIcon}</span>${originalIndex + 1}</td>
                <td class="truncated-text">${escapeHtml(textDisplay)}</td>
                <td><span class="count-badge">${matchCount} items</span></td>
                <td><span class="count-badge">${excludeCount} items</span></td>
                <td class="pattern-cell">${escapeHtml(patternDisplay)}</td>
                <td>
                    <button class="action-btn delete-btn" data-index="${originalIndex}" title="Delete">🗑️</button>
                </td>
            </tr>
            ${detailHtml}
        `;
    }).join('');

    datasetTableContainer.innerHTML = `
        <table class="dataset-table">
            <thead><tr>${headerHtml}</tr></thead>
            <tbody>${rowsHtml}</tbody>
        </table>
    `;

    // Add sort handlers
    datasetTableContainer.querySelectorAll('th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const col = th.dataset.sort;
            if (datasetSortColumn === col) {
                datasetSortDirection = datasetSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                datasetSortColumn = col;
                datasetSortDirection = 'asc';
            }
            renderDatasetTable();
        });
    });

    // Add delete handlers
    datasetTableContainer.querySelectorAll('.delete-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation(); // Don't trigger row expansion
            const index = parseInt(btn.dataset.index, 10);
            showDeleteConfirmModal(index);
        });
    });

    // Add expandable row handlers
    datasetTableContainer.querySelectorAll('.expandable-row').forEach(row => {
        row.addEventListener('click', (e) => {
            if (e.target.closest('.action-btn')) return;

            const rowId = row.dataset.rowId;
            if (expandedRows.has(rowId)) {
                expandedRows.delete(rowId);
            } else {
                expandedRows.add(rowId);
            }
            renderDatasetTable();
        });
    });
}

/**
 * Show delete confirmation modal.
 * @param {number} index - Index of dataset item to delete
 */
function showDeleteConfirmModal(index) {
    const example = currentDataset[index];

    modalTitle.textContent = 'Delete Training Example?';
    modalContent.innerHTML = `
        <p>Are you sure you want to delete this training example?</p>
        <div class="preview-item">
            <div class="preview-label">Pattern</div>
            <div class="preview-value pattern">${escapeHtml(example.expected_pattern || '')}</div>
        </div>
    `;
    modalConfirm.textContent = 'Delete';

    modalConfirmCallback = () => {
        if (typeof ahk !== 'undefined' && ahk.deleteFromDataset) {
            ahk.deleteFromDataset(index);
        }
    };

    showModal();
}

// Dataset search handler
datasetSearch.addEventListener('input', () => {
    renderDatasetTable();
});

// Refresh dataset button
refreshDatasetBtn.addEventListener('click', () => {
    if (typeof ahk !== 'undefined' && ahk.loadDataset) {
        datasetTableContainer.innerHTML = '<div class="dataset-loading">Loading dataset...</div>';
        ahk.loadDataset();
    }
});

// ===========================================
// CONFIG TAB
// ===========================================

/**
 * Get current config from form inputs.
 * @returns {Object} Config object
 */
function getSessionConfig() {
    return {
        model: configModel.value,
        temperature: parseFloat(configTemperature.value),
        max_attempts: parseInt(configMaxAttempts.value, 10),
        reward_threshold: parseFloat(configRewardThreshold.value),
        weights: {
            matches_all: parseFloat(weightInputs.matches_all.value),
            excludes_all: parseFloat(weightInputs.excludes_all.value),
            coherence: parseFloat(weightInputs.coherence.value),
            generalization: parseFloat(weightInputs.generalization.value),
            simplicity: parseFloat(weightInputs.simplicity.value)
        }
    };
}

/**
 * Update weight bars and total display.
 */
function updateWeightDisplay() {
    let total = 0;

    Object.keys(weightInputs).forEach(key => {
        const value = parseFloat(weightInputs[key].value) || 0;
        total += value;

        // Update the bar next to this input
        const bar = weightInputs[key].parentElement.querySelector('.weight-fill');
        if (bar) {
            bar.style.width = `${value * 100}%`;
        }
    });

    // Update total display
    weightTotal.textContent = `(${total.toFixed(2)})`;
    weightTotal.className = 'weight-total ' + (Math.abs(total - 1.0) < 0.01 ? 'valid' : 'invalid');
}

/**
 * Reset config to defaults.
 */
function resetConfig() {
    configModel.value = defaultConfig.model;
    configTemperature.value = defaultConfig.temperature;
    configMaxAttempts.value = defaultConfig.max_attempts;
    configRewardThreshold.value = defaultConfig.reward_threshold;

    Object.keys(weightInputs).forEach(key => {
        weightInputs[key].value = defaultConfig.weights[key];
    });

    updateWeightDisplay();
}

// Config input change handlers
Object.values(weightInputs).forEach(input => {
    input.addEventListener('input', updateWeightDisplay);
});

resetConfigBtn.addEventListener('click', resetConfig);

// ===========================================
// MODAL SYSTEM
// ===========================================

/**
 * Show the modal overlay.
 */
function showModal() {
    modalOverlay.style.display = '';
}

/**
 * Hide the modal overlay.
 */
function hideModal() {
    modalOverlay.style.display = 'none';
    modalConfirmCallback = null;
}

// Modal event handlers
modalCancel.addEventListener('click', hideModal);
modalClose.addEventListener('click', hideModal);
modalOverlay.addEventListener('click', (e) => {
    if (e.target === modalOverlay) hideModal();
});

modalConfirm.addEventListener('click', () => {
    if (modalConfirmCallback) {
        modalConfirmCallback();
    }
    hideModal();
});

// ===========================================
// UTILITY FUNCTIONS (Additional)
// ===========================================

/**
 * Truncate text to a maximum length.
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated text
 */
function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// ===========================================
// INITIALIZATION
// ===========================================

// Set initial UI state
updateUI();
updateWeightDisplay();

// Clear Results button handler
const clearResultsBtn = document.getElementById('clearResults');
if (clearResultsBtn) {
    clearResultsBtn.addEventListener('click', function() {
        currentResults = null;
        addedToDataset = new Set();
        expandedRows = new Set();
        renderResultsTable();
    });
}
