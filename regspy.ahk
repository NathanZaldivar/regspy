#Requires AutoHotkey v2.0
#SingleInstance Force

; ===========================================
; REGEX GENERATOR - AutoHotkey WebView2 Application
;
; A GUI tool for generating regex patterns from highlighted text.
; Uses a local Ollama LLM via regspy.py for pattern generation.
;
; Features:
;   - Live clipboard monitoring and display
;   - Text selection highlighting (positive examples - cyan)
;   - Right-click to mark as exclude (negative examples - red)
;   - LLM generates patterns matching positives, avoiding negatives
;   - Frameless, always-on-top window
;
; Dependencies:
;   - WebViewToo.ahk library
;   - JSON.ahk library
;   - Python 3.x with ollama, pydantic packages
;   - regspy.py in same directory
; ===========================================

; ===========================================
; INCLUDES
; ===========================================

; WebView2 wrapper library for embedding Chromium browser
#Include lib/WebViewToo.ahk

; JSON parsing library for data interchange
#Include lib/JSON.ahk

; ===========================================
; CONFIGURATION
; ===========================================

; Path to Python executable (adjust if not in PATH)
global PYTHON_PATH := "python"

; Temp directory for input/output JSON files
global TEMP_DIR := A_Temp

; ===========================================
; GLOBALS
; ===========================================

; Main WebView window reference
global MyWindow := ""

; Flag to track if page has finished loading
global PageLoaded := false

; Async process tracking
global PendingProcess := Map()  ; Stores output file path, input file, config file, PID

; ===========================================
; MAIN ENTRY POINT
; ===========================================

/**
 * Initialize and launch the Regex Generator application.
 * Sets up WebView2 window, registers callbacks, and starts clipboard monitoring.
 */
Main() {
    global MyWindow, PageLoaded
    
    ; Get directory where this script lives (for relative paths)
    SplitPath(A_LineFile,, &scriptDir)
    
    ; WebView2 configuration
    ; DataDir: where WebView2 stores profile data
    ; DllPath: path to WebView2Loader.dll
    WebViewSettings := {
        DataDir: WebViewCtrl.TempDir,
        DllPath: "WebView2Loader.dll"
    }
    
    ; Create frameless, always-on-top window
    ; +Resize: allow window resizing
    ; -Caption: remove title bar (frameless)
    ; +AlwaysOnTop: keep window above others
    MyWindow := WebViewGui("+Resize -Caption +AlwaysOnTop",,, WebViewSettings)
    
    ; Register window close handler
    MyWindow.OnEvent("Close", CleanupAndClose)
    
    ; ===========================================
    ; JAVASCRIPT -> AHK CALLBACKS
    ; These functions can be called from JS via: ahk.functionName(args)
    ; ===========================================

    ; Called when user clicks "Generate Regex" button
    MyWindow.AddCallbackToScript("generateRegex", GenerateRegex)

    ; Called when user clicks "Cancel" button
    MyWindow.AddCallbackToScript("cancel", CancelWindow)

    ; Called to load the training dataset
    MyWindow.AddCallbackToScript("loadDataset", LoadDataset)

    ; Called to add a result to the training dataset
    MyWindow.AddCallbackToScript("addToDataset", AddToDataset)

    ; Called to delete an item from the training dataset
    MyWindow.AddCallbackToScript("deleteFromDataset", DeleteFromDataset)
    
    ; Register navigation complete handler to know when page is ready
    MyWindow.wv.add_NavigationCompleted(OnNavComplete)

    ; Navigate to the HTML interface
    MyWindow.Navigate(scriptDir "\regspy.html")
    
    ; Show window (w600 h500 = 600px wide, 500px tall)
    MyWindow.Show("w600 h500")
}

; ===========================================
; NAVIGATION HANDLERS
; ===========================================

/**
 * Called when WebView2 finishes loading the page.
 *
 * @param sender - WebView2 sender object
 * @param args - Navigation completed event args
 */
OnNavComplete(sender, args) {
    global PageLoaded

    ; Mark page as loaded
    PageLoaded := true
}

/**
 * Encode a string to base64 using Windows Crypt API.
 * 
 * @param str - String to encode
 * @returns Base64 encoded string (no CRLF line breaks)
 */
Base64Encode(str) {
    ; Flag for base64 without CRLF line breaks
    static CRYPT_STRING_BASE64_NOCRLF := 0x40000001
    
    ; Handle empty string
    if !str
        return ""
    
    ; Convert string to UTF-8 bytes
    buf := Buffer(StrPut(str, "UTF-8") - 1)
    StrPut(str, buf, "UTF-8")
    
    ; First call: get required output size
    DllCall("crypt32\CryptBinaryToStringW", 
        "Ptr", buf, 
        "UInt", buf.Size, 
        "UInt", CRYPT_STRING_BASE64_NOCRLF, 
        "Ptr", 0, 
        "UInt*", &size := 0)
    
    ; Allocate output buffer
    out := Buffer(size * 2)
    
    ; Second call: perform encoding
    DllCall("crypt32\CryptBinaryToStringW", 
        "Ptr", buf, 
        "UInt", buf.Size, 
        "UInt", CRYPT_STRING_BASE64_NOCRLF, 
        "Ptr", out, 
        "UInt*", &size)
    
    return StrGet(out)
}

; ===========================================
; REGEX GENERATION
; ===========================================

/**
 * Generate regex patterns using the Python LLM script.
 * Called from JS when user clicks "Generate Regex" button.
 *
 * Flow:
 *   1. Create input JSON with text, highlighted items, and excluded items
 *   2. Write to temp file
 *   3. Run regspy.py asynchronously with input/output file paths
 *   4. Poll for output file completion
 *   5. Read output JSON and send to JS
 *
 * @param WebView - WebView2 reference (auto-passed by AddCallbackToScript)
 * @param fullText - Original clipboard text
 * @param selectionsJson - JSON array string of selected text items to match
 * @param excludedJson - JSON array string of selected text items to exclude
 * @param configJson - JSON string of session config (optional)
 */
GenerateRegex(WebView, fullText, selectionsJson, excludedJson := "[]", configJson := "") {
    global MyWindow, PYTHON_PATH, TEMP_DIR, PendingProcess

    ; Cancel any pending process first
    CancelPendingProcess()

    ; Get script directory for regspy.py path
    SplitPath(A_LineFile,, &scriptDir)

    ; Parse selections JSON array
    try {
        selections := JSON.Load(selectionsJson)
    } catch {
        ShowError("Failed to parse selections")
        return
    }

    ; Parse excluded JSON array
    try {
        excluded := JSON.Load(excludedJson)
    } catch {
        excluded := []
    }

    ; Validate we have selections
    if (!selections || selections.Length = 0) {
        ShowError("No text selected")
        return
    }

    ; ===========================================
    ; CREATE INPUT JSON
    ; Format: {"text": "...", "Highlighted Items": [...], "Excluded Items": [...]}
    ; ===========================================

    inputData := Map()
    inputData["text"] := fullText
    inputData["Highlighted Items"] := selections
    inputData["Excluded Items"] := excluded

    ; Generate unique filenames using timestamp
    timestamp := A_TickCount
    inputFile := TEMP_DIR "\regspy_input_" timestamp ".json"
    outputFile := TEMP_DIR "\regspy_output_" timestamp ".json"
    configFile := ""

    ; Write input JSON to temp file
    try {
        inputJson := JSON.Dump(inputData)
        FileAppend(inputJson, inputFile, "UTF-8-RAW")
    } catch as e {
        ShowError("Failed to write input file: " e.Message)
        return
    }

    ; Write config file if provided
    if (configJson != "") {
        configFile := TEMP_DIR "\regspy_config_" timestamp ".json"
        try {
            FileAppend(configJson, configFile, "UTF-8-RAW")
        } catch as e {
            ShowError("Failed to write config file: " e.Message)
            return
        }
    }

    ; ===========================================
    ; RUN PYTHON SCRIPT ASYNCHRONOUSLY
    ; ===========================================

    ; Build command line
    pythonScript := scriptDir "\regspy.py"
    cmdLine := PYTHON_PATH ' "' pythonScript '" "' inputFile '" "' outputFile '"'

    ; Add config flag if we have a config file
    if (configFile != "") {
        cmdLine .= ' --config "' configFile '"'
    }

    ; Show loading state in UI
    MyWindow.ExecuteScriptAsync('setLoading(true)')

    ; Store pending process info for the polling callback
    PendingProcess["outputFile"] := outputFile
    PendingProcess["inputFile"] := inputFile
    PendingProcess["configFile"] := configFile
    PendingProcess["startTime"] := A_TickCount

    ; Run Python script asynchronously (don't wait)
    try {
        Run(cmdLine,, "Hide", &pid)
        PendingProcess["pid"] := pid
    } catch as e {
        ShowError("Failed to run Python: " e.Message)
        CleanupTempFiles(inputFile, outputFile, configFile)
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        PendingProcess.Clear()
        return
    }

    ; Start polling for completion (every 200ms)
    SetTimer(CheckProcessComplete, 200)
}

/**
 * Poll for Python process completion.
 * Checks if output file exists and process has finished.
 */
CheckProcessComplete() {
    global MyWindow, PendingProcess

    ; Safety check - no pending process
    if (!PendingProcess.Has("outputFile")) {
        SetTimer(CheckProcessComplete, 0)  ; Stop timer
        return
    }

    outputFile := PendingProcess["outputFile"]
    inputFile := PendingProcess["inputFile"]
    configFile := PendingProcess.Has("configFile") ? PendingProcess["configFile"] : ""
    pid := PendingProcess.Has("pid") ? PendingProcess["pid"] : 0
    startTime := PendingProcess["startTime"]

    ; Check for timeout (60 seconds)
    if (A_TickCount - startTime > 60000) {
        SetTimer(CheckProcessComplete, 0)  ; Stop timer
        if (pid)
            ProcessClose(pid)
        ShowError("Generation timed out after 60 seconds")
        CleanupTempFiles(inputFile, outputFile, configFile)
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        PendingProcess.Clear()
        return
    }

    ; Check if process is still running
    if (pid && ProcessExist(pid)) {
        ; Still running, continue polling
        return
    }

    ; Process finished, stop polling
    SetTimer(CheckProcessComplete, 0)

    ; ===========================================
    ; READ OUTPUT JSON
    ; ===========================================

    ; Check if output file exists
    if (!FileExist(outputFile)) {
        ShowError("Python script did not produce output")
        CleanupTempFiles(inputFile, outputFile, configFile)
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        PendingProcess.Clear()
        return
    }

    ; Read output JSON
    try {
        outputJson := FileRead(outputFile, "UTF-8-RAW")
        result := JSON.Load(outputJson)
    } catch as e {
        ShowError("Failed to read output: " e.Message)
        CleanupTempFiles(inputFile, outputFile, configFile)
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        PendingProcess.Clear()
        return
    }

    ; Cleanup temp files
    CleanupTempFiles(inputFile, outputFile, configFile)
    PendingProcess.Clear()

    ; ===========================================
    ; HANDLE RESULT
    ; ===========================================

    ; Check for error in result
    if (result.Has("error")) {
        ShowError(result["error"])
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        return
    }

    ; Check for results array (new format)
    if (!result.Has("results") || result["results"].Length = 0) {
        ShowError("No patterns generated")
        MyWindow.ExecuteScriptAsync('setLoading(false)')
        return
    }

    ; Send full result object to JS (includes results array with detailed scoring)
    ; onRegexGenerated() is defined in regspy.js
    resultsJson := JSON.Dump(result)

    ; Escape for JS string (order matters: backslashes first, then quotes, then newlines)
    resultsJson := StrReplace(resultsJson, "\", "\\")
    resultsJson := StrReplace(resultsJson, "'", "\'")
    resultsJson := StrReplace(resultsJson, "`n", "\n")
    resultsJson := StrReplace(resultsJson, "`r", "\r")

    MyWindow.ExecuteScriptAsync("onRegexGenerated('" resultsJson "')")
    MyWindow.ExecuteScriptAsync('setLoading(false)')
}

/**
 * Cancel any pending Python process.
 */
CancelPendingProcess() {
    global PendingProcess

    if (PendingProcess.Count = 0)
        return

    ; Stop the polling timer
    SetTimer(CheckProcessComplete, 0)

    ; Kill the process if running
    if (PendingProcess.Has("pid")) {
        pid := PendingProcess["pid"]
        if (ProcessExist(pid))
            ProcessClose(pid)
    }

    ; Cleanup temp files
    if (PendingProcess.Has("inputFile") && PendingProcess.Has("outputFile")) {
        CleanupTempFiles(
            PendingProcess["inputFile"],
            PendingProcess["outputFile"],
            PendingProcess.Has("configFile") ? PendingProcess["configFile"] : ""
        )
    }

    PendingProcess.Clear()
}

/**
 * Display an error message in the UI.
 * 
 * @param msg - Error message to display
 */
ShowError(msg) {
    global MyWindow
    
    ; Escape for JS string
    msg := StrReplace(msg, "\", "\\")
    msg := StrReplace(msg, "'", "\'")
    msg := StrReplace(msg, "`n", "\n")
    
    MyWindow.ExecuteScriptAsync("showError('" msg "')")
}

/**
 * Delete temporary input/output/config files.
 *
 * @param inputFile - Path to input JSON file
 * @param outputFile - Path to output JSON file
 * @param configFile - Path to config JSON file (optional)
 */
CleanupTempFiles(inputFile, outputFile, configFile := "") {
    try {
        if (FileExist(inputFile))
            FileDelete(inputFile)
        if (FileExist(outputFile))
            FileDelete(outputFile)
        if (configFile != "" && FileExist(configFile))
            FileDelete(configFile)
    }
}

; ===========================================
; DATASET MANAGEMENT
; ===========================================

/**
 * Load the training dataset from Python.
 * Called when user visits the Dataset tab.
 *
 * @param WebView - WebView2 reference (auto-passed by AddCallbackToScript)
 */
LoadDataset(WebView) {
    global MyWindow, PYTHON_PATH, TEMP_DIR

    ; Get script directory for regspy.py path
    SplitPath(A_LineFile,, &scriptDir)

    ; Generate unique output filename
    timestamp := A_TickCount
    outputFile := TEMP_DIR "\regspy_dataset_" timestamp ".json"

    ; Build command line
    pythonScript := scriptDir "\regspy.py"
    cmdLine := PYTHON_PATH ' "' pythonScript '" --list-dataset "' outputFile '"'

    ; Run Python script
    try {
        results := SimpleExec(cmdLine)
    } catch as e {
        ShowError("Failed to load dataset: " e.Message)
        return
    }

    ; Check if output file exists
    if (!FileExist(outputFile)) {
        ShowError("Failed to load dataset")
        return
    }

    ; Read output JSON
    try {
        outputJson := FileRead(outputFile, "UTF-8-RAW")
    } catch as e {
        ShowError("Failed to read dataset: " e.Message)
        FileDelete(outputFile)
        return
    }

    ; Cleanup temp file
    FileDelete(outputFile)

    ; Escape for JS string (order matters: backslashes first, then quotes, then newlines)
    outputJson := StrReplace(outputJson, "\", "\\")
    outputJson := StrReplace(outputJson, "'", "\'")
    outputJson := StrReplace(outputJson, "`n", "\n")
    outputJson := StrReplace(outputJson, "`r", "\r")

    ; Send to JS
    MyWindow.ExecuteScriptAsync("onDatasetLoaded('" outputJson "')")
}

/**
 * Add an example to the training dataset.
 * Called when user confirms adding a result to the dataset.
 *
 * @param WebView - WebView2 reference (auto-passed by AddCallbackToScript)
 * @param exampleJson - JSON string of the example to add
 * @param resultIndex - Index of the result being added (for UI feedback)
 */
AddToDataset(WebView, exampleJson, resultIndex) {
    global MyWindow, PYTHON_PATH, TEMP_DIR

    ; Get script directory for regspy.py path
    SplitPath(A_LineFile,, &scriptDir)

    ; Generate unique input filename
    timestamp := A_TickCount
    inputFile := TEMP_DIR "\regspy_example_" timestamp ".json"

    ; Write example to temp file
    try {
        FileAppend(exampleJson, inputFile, "UTF-8-RAW")
    } catch as e {
        ShowError("Failed to write example: " e.Message)
        return
    }

    ; Build command line
    pythonScript := scriptDir "\regspy.py"
    cmdLine := PYTHON_PATH ' "' pythonScript '" --add-example "' inputFile '"'

    ; Run Python script
    try {
        results := SimpleExec(cmdLine)
    } catch as e {
        ShowError("Failed to add to dataset: " e.Message)
        FileDelete(inputFile)
        return
    }

    ; Cleanup temp file
    FileDelete(inputFile)

    ; Notify JS of success
    MyWindow.ExecuteScriptAsync("onExampleAdded(" resultIndex ")")
}

/**
 * Delete an example from the training dataset.
 * Called when user confirms deleting a dataset item.
 *
 * @param WebView - WebView2 reference (auto-passed by AddCallbackToScript)
 * @param index - Index of the example to delete
 */
DeleteFromDataset(WebView, index) {
    global MyWindow, PYTHON_PATH

    ; Get script directory for regspy.py path
    SplitPath(A_LineFile,, &scriptDir)

    ; Build command line
    pythonScript := scriptDir "\regspy.py"
    cmdLine := PYTHON_PATH ' "' pythonScript '" --delete-example ' index

    ; Run Python script
    try {
        results := SimpleExec(cmdLine)
    } catch as e {
        ShowError("Failed to delete from dataset: " e.Message)
        return
    }

    ; Notify JS of success (will trigger dataset refresh)
    MyWindow.ExecuteScriptAsync("onDatasetItemDeleted(" index ")")
}

; ===========================================
; WINDOW MANAGEMENT
; ===========================================

/**
 * Cancel/hide the window without action.
 * Called from JS when user clicks "Cancel" button.
 *
 * @param WebView - WebView2 reference (optional, auto-passed)
 */
CancelWindow(WebView := "") {
    global MyWindow

    ; Cancel any pending generation
    CancelPendingProcess()

    if IsSet(MyWindow) && MyWindow {
        MyWindow.Hide()
    }
}

/**
 * Clean up resources and exit application.
 * Called when window is closed.
 *
 * @param thisGui - GUI reference (optional)
 */
CleanupAndClose(thisGui := "") {
    global MyWindow

    ; Cancel any pending generation
    CancelPendingProcess()

    ; Destroy window and clean up
    if IsSet(MyWindow) {
        MyWindow.Destroy()
        MyWindow := ""
    }

    ExitApp()
}

SimpleExec(command) {

    ; Why Exec doesn't let you hide the CMD? WHO KNOWS! :D
    ; Why Run doesn't let you retrieve output without having
    ; to do the stupid work around with a temp file? WHO KNOWS :D
    ; https://stackoverflow.com/questions/15820253/capture-cmd-output-with-autohotkey

    DetectHiddenWindows(1)
    Run(A_ComSpec,, "Hide", &pid)
    WinWait("ahk_pid" pid)
    DllCall("AttachConsole", "UInt", pid)
  
    shell := ComObject("WScript.Shell")
    exec := shell.Exec(A_ComSpec " /C " command)
    result := exec.StdOut.ReadAll()
   
    DllCall("FreeConsole")
    ProcessClose(pid)
    return result
}

; ===========================================
; AUTO-START
; Launch the application when script runs
; ===========================================

Main()