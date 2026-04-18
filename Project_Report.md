# Project Report: TensorScript Neural Network Compiler & IDE

## 1. Abstract
The development of Deep Learning networks often requires significant boilerplate code within frameworks like PyTorch or TensorFlow natively. This project introduces **TensorScript**, a minimal, declarative Domain-Specific Language (DSL) engineered explicitly for defining model architectures seamlessly. To support TensorScript, we fully designed, implemented, and executed a custom 7-Phase Compiler from scratch explicitly in Python. The compiler translates raw `.ts` text logically into functional PyTorch python modules via Mathematical Abstract Syntax Trees. Furthermore, a standalone Desktop Graphical User Interface (GUI) reflecting a nostalgic 1990s retro aesthetic was securely integrated to provide local development capabilities entirely offline.

## 2. Introduction & Objective
**Problem Statement:** Configuring layers, ensuring accurate matrix alignment natively, and setting up training loops structurally in native ML ecosystems can be verbose and prone to silent topological compilation errors natively.
**Objective:** To design a custom language abstracting these complexities internally, alongside a compiler enforcing topological Machine Learning rules semantically. This ensures that only mathematically sound models actively generate corresponding PyTorch codes natively prior to execution iteratively.

## 3. Language Design & Syntax
TensorScript is statically typed sequentially utilizing bracketed scoping formats explicitly. It supports four core domain principles:
*   **Dimensionality Configurations**: `input shape(N)`
*   **Layer Construction**: `layer [type](parameters, activation)` 
*   **Compilation Strategy**: `loss [type]` and `optimizer [type](lr=N)`
*   **Execution Routines**: `train epochs=N, batch=N`

**Example DSL Generation:**
```typescript
model BinaryPredictor {
  input shape(256)
  layer dense(64, relu)
  layer dropout(0.5)
  layer dense(1, sigmoid)
  loss binary_crossentropy
  optimizer sgd(lr=0.01)
  train epochs=15, batch=64
}
```

## 4. The 7-Phase Compiler Architecture
The core system avoids external grammar dependencies natively (like ANTLR) executing entirely sequentially across 7 unique Python module phases accurately:

*   **Phase 1: Lexical Analysis (`lexer.py`)** 
    Utilizes Regular Expression capture groups matching sequence strings exactly to isolate keywords logically (e.g. `dense`, `dropout`, `adam`) into validated system Tokens dynamically.
*   **Phase 2: Recursion Parsing (`parser.py`)** 
    Transforms flat Arrays of Tokens back into a structurally enforced Abstract Syntax Tree (AST). It enforces grammatical sequential relationships (ensuring inputs always evaluate before layer definitions structurally).
*   **Phase 3: Semantic Analysis (`semantic.py`)** 
    Validates machine learning topology natively. Evaluates rule conditions structurally (e.g. `categorical_crossentropy` explicitly requires a `softmax` activation explicitly, flagging custom Semantic Validation exceptions dynamically otherwise).
*   **Phase 4: Intermediate Representation Generator (`ir.py`)** 
    Flattens the massive visual AST tree dynamically into linear 3-Address Code computing instructions logically tracing mathematical pointer allocations cleanly (e.g. `%1 = dense %0`, `%2 = relu %1`).
*   **Phase 5: Computational Optimizer (`optimizer.py`)** 
    Intercepts the logic block applying recursive mathematical shortcuts internally including:
    - **Dead Code Elimination**: Purging any mathematical branches unassigned to final outputs explicitly.
    - **Constant Folding**: Canceling nullified logic traces (e.g. `dropout(0.0)`) skipping structural GPU delays dynamically.
*   **Phase 6: Code Generation (`codegen.py`)** 
    Analyzes final validated IR variables mapping them flawlessly to object-oriented Native `PyTorch` equivalents physically emitting fully operational strings of raw native Python files (`generated_model.py`) to the local OS exactly matching standard machine-learning formats.
*   **Phase 7: Logic Execution Hook (`compiler.py`)** 
    Unifies the 6 internal phases logically executing trace streams directly over local memory securely triggering validation pipelines structurally.

## 5. Desktop Application Studio
An isolated **Native GUI IDE** (`gui.py`) was mapped securely using Python's raw `tkinter` module specifically styled natively after standard MS-DOS / Windows 95 aesthetics explicitly.
- **Design Parameters:** Utilizes `Courier New` Hacker-Green textual contrasts over stark `#000000` Black matrix templates matching classic hacker topological interfaces effortlessly to enhance developer interaction visually. 
- **Sys Hooking:** The application hijacks standard memory allocations bridging inputs/outputs structurally over `io.StringIO` directly capturing generated PyTorch code blocks safely in local interface windows avoiding dangerous network transmission protocols natively.

## 6. Implementation Results
The execution of TensorScript logically yielded successful real-time Native PyTorch validation loops iteratively evaluating matrices cleanly! The trace log actively confirms proper Epoch convergence dynamically ensuring loss calculations calculate mathematically optimally.

## 7. Future Scope & Extensibility
The modular architecture allows for infinite expandability structurally! Future developments natively involve:
- Structurally bridging Convolutional Networks (`Conv2d`) inherently adapting parameters logically for computer vision image modeling.
- Establishing multiple backend targets explicitly allowing output generations compiling dynamically into `TensorFlow` or `ONNX` mathematically!
