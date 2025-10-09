# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
- **Install package in development mode**: `pip install -e ".[dev]"`
- **Install with all providers**: `pip install -e ".[all]"`
- **Install specific provider**: `pip install -e ".[openai]"` or `pip install -e ".[anthropic]"`
- **Run demo**: `compositional-demo` or `python -m compositional_prompting.examples.demo`

### Testing & Quality
- **Run all tests**: `pytest`
- **Run specific test**: `pytest path/to/test.py::test_function_name`
- **Run tests with coverage**: `pytest --cov=compositional_prompting --cov-report=term-missing`
- **Run tests with verbose output**: `pytest -v`
- **Format code**: `black compositional_prompting/ --line-length 100`
- **Check formatting**: `black compositional_prompting/ --check`
- **Type checking**: `mypy compositional_prompting/ --strict`
- **Linting**: `flake8 compositional_prompting/ --max-line-length=100`

### Build & Distribution
- **Build package**: `python -m build`
- **Check package**: `twine check dist/*`
- **Upload to PyPI Test**: `twine upload --repository testpypi dist/*`
- **Upload to PyPI**: `twine upload dist/*`

## Architecture Overview

This is a Python package for compositional prompting - a framework for building sophisticated LLM prompts through discrete cognitive operations. The library provides a systematic approach to prompt engineering using a fluid API.

### Core Design Philosophy
The framework is built around **compositional action space** that factors prompts into five orthogonal dimensions:
- **ω (Cognitive Operation)**: DECOMPOSE, ANALYZE, GENERATE, VERIFY, SYNTHESIZE, ABSTRACT - The primary thinking mode
- **φ (Focus Aspect)**: STRUCTURE, CONSTRAINTS, PATTERNS, SOLUTION, CORRECTNESS, RELATIONSHIPS, ALTERNATIVE_APPROACHES, EDGE_CASES - What to focus on
- **σ (Reasoning Style)**: SYSTEMATIC, CREATIVE, CRITICAL, FORMAL, INTUITIVE - How to approach the problem
- **κ (Connection Type)**: THEREFORE, HOWEVER, BUILDING_ON, ALTERNATIVELY - How to connect reasoning steps
- **τ (Output Format)**: STEPS, LIST, MATHEMATICAL, NARRATIVE, CODE, SOLUTION - How to structure the output

This factorization reduces complexity from 30,720 possible prompt combinations to 64 manageable parameters while maintaining full expressiveness through composition.

### Key Components

#### Core Classes (`compositional_prompting/__init__.py`)
- **`ComposingPrompt`**: Main fluid API class that builds prompts through method chaining
  - Maintains internal state for problem context, cognitive operations, and LLM augmentations
  - Supports action vector analysis for interpretability
  - Provides weighted sampling for biased exploration
- **`LLMProvider`**: Abstract base class defining the provider interface
  - `generate()`: Core generation method
  - `get_provider_name()`: Provider identification
- **Provider Implementations**:
  - `MockLLMProvider`: Testing without API calls
  - `OpenAIProvider`: GPT-3.5/4 integration
  - `AnthropicProvider`: Claude integration
  - `OllamaProvider`: Local model support
- **Enum Classes**: Type-safe compositional dimensions
  - `CognitiveOperation`, `FocusAspect`, `ReasoningStyle`, `ConnectionType`, `OutputFormat`

#### Method Chaining Pattern
The fluid API allows natural prompt construction:
```python
prompt.cognitive_op().focus().style().connection().output_format().build()
```

#### LLM Augmentation Methods
- `llm_augment()`: Add reasoning from LLM
- `llm_add_examples()`: Generate examples (supports parallel execution)
- `llm_coherence_check()`: Verify logical consistency
- `llm_termination()`: Smart completion detection
- `llm_evaluate()`: Score reasoning quality

### Package Structure
```
compositional_prompting/
├── __init__.py           # All core functionality (single-file architecture)
│                         # - LLMProvider abstract class and implementations
│                         # - ComposingPrompt fluid API
│                         # - Enums for compositional dimensions
│                         # - Weighted sampling and orchestration
└── examples/
    ├── __init__.py      # Empty module file
    └── demo.py          # Interactive demonstrations of all features
```

**Note**: The library uses a deliberate single-file architecture where all core functionality lives in `__init__.py`. This design choice simplifies maintenance and distribution while keeping the codebase accessible.

### Integration Patterns

#### MCTS Integration
The library is designed as the action space for Monte Carlo Tree Search reasoning:
```python
# Each MCTS node represents a compositional action
action = ComposingPrompt.sample_action(weights=learned_priors)
vector = action.get_action_vector()  # For MCTS state representation
```

#### Multi-Agent Orchestration
Supports parallel execution across multiple providers:
```python
prompt.execute_llm_pipeline(providers=[gpt4, claude], max_workers=4)
```

#### Custom Provider Integration
Extend `LLMProvider` for any LLM backend:
```python
class CustomProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        # Your implementation
```

### Development Notes

#### Python Version & Dependencies
- **Minimum Python**: 3.8+ (uses typing features from 3.8)
- **Core dependency**: `typing-extensions>=4.0.0` for backward compatibility
- **Optional providers**: Install with `pip install -e ".[openai]"` or `".[anthropic]"`

#### Code Quality Settings
- **Black formatter**: Line length = 100 characters
- **MyPy**: Strict mode enabled (`disallow_untyped_defs=true`)
- **Target Python versions**: 3.8, 3.9, 3.10, 3.11

#### Testing Approach
- Tests should cover all compositional dimensions
- Mock provider for testing without API calls
- Parallel execution tests with `ThreadPoolExecutor`
- Currently no test files exist - tests should be added to `tests/` directory

#### Entry Points
- Console script: `compositional-demo` (defined in pyproject.toml)
- Module execution: `python -m compositional_prompting.examples.demo`