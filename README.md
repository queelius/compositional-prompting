# Compositional Prompting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A standalone library that provides a structured compositional action space for LLM prompting. This framework enables external systems (MCTS, RL agents, evolutionary algorithms, etc.) to systematically explore and construct prompts through discrete, composable cognitive operations.

**Important**: This is NOT an MCTS or RL implementation. It's a compositional framework that such systems can use as their action space.

## Key Features

🧩 **Compositional Action Space**: Factor prompts into orthogonal dimensions (ω,φ,σ,κ,τ)
🎲 **Structured Sampling**: Sample actions uniformly or with learned/manual weights
📊 **Action Vectors**: Convert prompts to vectors for ML model integration
🔗 **Compositional Algebra**: Combine prompts using compose, chain, and fork operations
🎯 **External System Ready**: Designed for MCTS, RL agents, and other controllers
🔌 **Multi-Provider Support**: Works with OpenAI, Anthropic, or any LLM  

## Installation

```bash
# Basic installation
pip install compositional-prompting

# With OpenAI support
pip install compositional-prompting[openai]

# With all providers
pip install compositional-prompting[all]
```

## Quick Start

### Basic Compositional Prompt

```python
from compositional_prompting import ComposingPrompt, CognitiveOperation, FocusAspect

# Build a structured reasoning prompt
prompt = (ComposingPrompt()
    .cognitive_op(CognitiveOperation.DECOMPOSE)
    .focus(FocusAspect.STRUCTURE) 
    .problem_context("Solve the optimization problem: maximize x² + y² subject to x + y = 10")
    .build())

print(prompt)
# Output: "Problem: Solve the optimization problem...
# Let me decompose this problem systematically. I'll focus on the structural relationships..."
```

### Weighted Action Sampling

```python
# Define reasoning preferences
weights = {
    'cognitive_op': {
        CognitiveOperation.DECOMPOSE: 3.0,  # Strongly prefer breaking down problems
        CognitiveOperation.VERIFY: 2.0,     # Emphasize verification  
        CognitiveOperation.ANALYZE: 1.0,    # Baseline analysis
    }
}

# Sample actions with bias
action = ComposingPrompt.sample_action(weights=weights)
vector = action.get_action_vector()
print(f"Sampled: ω={vector['omega']}, φ={vector['phi']}")
# More likely to sample DECOMPOSE due to higher weight
```

### Multi-Provider Orchestration

```python
from compositional_prompting import OpenAIProvider, AnthropicProvider

# Use different models for different operations
fast_provider = OpenAIProvider(model="gpt-3.5-turbo")
slow_provider = OpenAIProvider(model="gpt-4")

prompt = (ComposingPrompt()
    .problem_context("Design a fraud detection system")
    .llm_add_examples(n=5, provider=fast_provider, parallel=True)    # Fast parallel
    .llm_augment("analyze security risks", provider=slow_provider)   # Slow sequential  
    .llm_coherence_check()  # Default provider
    .build())
```

### Smart Termination Detection

```python
# Combine pattern matching + LLM reasoning
prompt = ComposingPrompt()

# Classical patterns: "therefore", "QED", "final answer", etc.
# + LLM assessment of reasoning completeness
is_done = prompt.llm_termination("Therefore, the answer is x=5, y=5. QED.")
print(is_done)  # True
```

## Compositional Action Space

The framework is built around five key dimensions:

- **ω (Cognitive Operation)**: DECOMPOSE, ANALYZE, GENERATE, VERIFY, SYNTHESIZE, ABSTRACT
- **φ (Focus Aspect)**: STRUCTURE, CONSTRAINTS, PATTERNS, SOLUTION, CORRECTNESS, etc.  
- **σ (Reasoning Style)**: SYSTEMATIC, CREATIVE, CRITICAL, FORMAL, INTUITIVE
- **κ (Connection Type)**: THEREFORE, HOWEVER, BUILDING_ON, ALTERNATIVELY
- **τ (Output Format)**: STEPS, LIST, MATHEMATICAL, NARRATIVE, CODE, SOLUTION

This factorization reduces complexity from 30,720 possible combinations to 64 manageable parameters while maintaining expressiveness.

## Advanced Usage

### Parallel Execution

```python
# Execute multiple operations concurrently
prompt = (ComposingPrompt()
    .cognitive_op(CognitiveOperation.GENERATE)
    .llm_add_examples(n=4, parallel=True)      # Concurrent example generation
    .execute_llm_pipeline(max_workers=4)       # Control parallelism
    .build())
```

### Action Vector Analysis

```python
action = (ComposingPrompt()
    .cognitive_op(CognitiveOperation.ANALYZE)
    .focus(FocusAspect.PATTERNS)
    .style(ReasoningStyle.SYSTEMATIC))

vector = action.get_action_vector()
print(vector)
# {
#   'omega': 'analyze', 'phi': 'patterns', 'sigma': 'systematic',
#   'llm_augmentations': 0, 'has_parallel_ops': False, ...  
# }
```

### Custom Providers

```python
from compositional_prompting import LLMProvider

class CustomProvider(LLMProvider):
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        # Your custom LLM integration
        return your_llm_call(prompt)
    
    def get_provider_name(self) -> str:
        return "CustomLLM"

# Use with compositional prompts
prompt = ComposingPrompt().set_llm_provider(CustomProvider())
```

## How External Systems Use This Library

This library provides the action space that external systems can explore:

- **🎯 MCTS**: Each node's action is a `ComposingPrompt`, sampled with learned weights
- **🤖 RL Agents**: Actions are compositional prompts, Q-values learned over action vectors
- **🧬 Evolutionary Algorithms**: Genomes are compositional actions that can mutate/crossover
- **🔬 Research**: Systematic exploration of prompt space with interpretable dimensions
- **👤 Human-in-the-Loop**: Interactive selection of compositional dimensions

## Examples

```bash
# Run basic demo
compositional-demo

# Or directly
python -m compositional_prompting.examples.demo
```

## Architecture

The library uses a clean separation of concerns:

```
compositional_prompting/
├── __init__.py           # Core classes and enums
├── providers/            # LLM provider implementations  
├── actions/              # Cognitive operation definitions
├── orchestration/        # Parallel execution logic
└── examples/             # Demo applications
```

## Integration

This package serves as a foundation for:

- [`reasoning-llm-policy`](https://github.com/queelius/reasoning-llm-policy): MCTS + compositional reasoning
- [`mcts-reasoning`](https://github.com/queelius/mcts-reasoning): Pure MCTS implementation
- Your own reasoning systems and AI applications

## Contributing

```bash
git clone https://github.com/queelius/compositional-prompting.git
cd compositional-prompting
pip install -e ".[dev]"

# Run tests
pytest

# Format code  
black compositional_prompting/
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{towell2024compositional,
  title={Compositional Prompting: A Fluid API Framework for LLM Reasoning},
  author={Towell, Alex and Fujinoki, Hiroshi and Gultepe, Eren},
  year={2024},
  url={https://github.com/queelius/compositional-prompting}
}
```