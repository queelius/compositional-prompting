# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Install package in development mode**: `pip install -e ".[dev]"`
- **Install with all providers**: `pip install -e ".[all]"`
- **Run demo**: `compositional-demo` or `python -m compositional_prompting.examples.demo`

### Testing & Quality
- **Run tests**: `pytest`
- **Run tests with coverage**: `pytest --cov=compositional_prompting`
- **Format code**: `black compositional_prompting/`
- **Type checking**: `mypy compositional_prompting/`
- **Linting**: `flake8 compositional_prompting/`

### Build & Distribution
- **Build package**: `python -m build`
- **Check package**: `twine check dist/*`

## Architecture Overview

This is a Python package for compositional prompting - a framework for building sophisticated LLM prompts through discrete cognitive operations. The library provides a systematic approach to prompt engineering using a fluid API.

### Core Design Philosophy
The framework is built around **compositional action space** with five key dimensions:
- **ω (Cognitive Operation)**: DECOMPOSE, ANALYZE, GENERATE, VERIFY, SYNTHESIZE, ABSTRACT
- **φ (Focus Aspect)**: STRUCTURE, CONSTRAINTS, PATTERNS, SOLUTION, CORRECTNESS, etc.
- **σ (Reasoning Style)**: SYSTEMATIC, CREATIVE, CRITICAL, FORMAL, INTUITIVE
- **κ (Connection Type)**: THEREFORE, HOWEVER, BUILDING_ON, ALTERNATIVELY
- **τ (Output Format)**: STEPS, LIST, MATHEMATICAL, NARRATIVE, CODE, SOLUTION

This reduces complexity from 30,720 possible combinations to 64 manageable parameters while maintaining expressiveness.

### Key Components

#### Core Classes (`__init__.py`)
- **`ComposingPrompt`**: Main fluid API class with method chaining for building prompts
- **`LLMProvider`**: Abstract base class for LLM integrations (OpenAI, Anthropic, Ollama, Mock)
- **Enum classes**: Define the compositional action space (CognitiveOperation, FocusAspect, etc.)

#### Provider Architecture
- Pluggable LLM provider system with built-in providers for OpenAI, Anthropic, and Ollama
- Mock provider for testing and development
- Provider factory pattern for configuration-based instantiation

#### Key Features
- **Weighted Sampling**: Bias exploration with learned or manual priors using probability distributions
- **Parallel Orchestration**: Execute embarrassingly parallel operations concurrently using ThreadPoolExecutor
- **Smart Termination**: Pattern matching + LLM reasoning for detecting complete reasoning states
- **Fluid API**: Chainable method calls for intuitive prompt construction

### Package Structure
```
compositional_prompting/
├── __init__.py           # Core classes, enums, and provider implementations
└── examples/
    └── demo.py          # Demonstration of API usage and features
```

### Integration Patterns
- Designed for use with MCTS reasoning systems and multi-agent AI
- Supports both synchronous and asynchronous execution patterns
- Extensible provider architecture for custom LLM integrations
- Action vector analysis for interpretability and debugging

### Development Notes
- Uses Python 3.8+ with typing-extensions for compatibility
- Black formatter with 100-character line length
- MyPy type checking with strict settings
- Optional dependencies for different LLM providers (openai, anthropic)
- Entry point script via `compositional-demo` command