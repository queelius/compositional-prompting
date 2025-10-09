# Compositional Prompting Enhancement Layer

## Overview

The Enhancement Layer is an elegant, optional extension to the compositional prompting library that adds intelligent augmentation capabilities while maintaining the purity and determinism of the core system. It follows the Unix philosophy of "do one thing well" and provides beautiful, composable APIs.

## Architecture

```
Pure Compositional Core (deterministic)
           ↓
    [Enhancement Layer]
           ↓
    Orchestrator (coordinator)
     /    |    |    \
   RAG  Fast  Refs  Templates
```

## Key Features

### 1. Graph-Based RAG for Examples
- **Complex Network Structure**: Examples form a graph with similarity-based edges
- **Community Detection**: Identifies clusters of related examples
- **Hub Detection**: Finds most connected, influential examples
- **Bridge Detection**: Identifies examples connecting different communities
- **Multiple Retrieval Strategies**:
  - `similar`: Pure similarity-based retrieval
  - `diverse`: Mix of similar and diverse examples
  - `hub_aware`: Prioritizes hub examples
  - `community`: Samples from different communities

### 2. Fast Local LLM Augmentation
- **Time-Bounded Operations**: All operations respect strict time limits (100-200ms)
- **Context Preparation**: Makes prompts specific without solving
- **Graceful Degradation**: Falls back to templates when time expires
- **Smart Caching**: Remembers augmentations for repeated queries
- **Pattern-Based Analysis**:
  - Key term extraction
  - Constraint identification
  - Decomposition suggestions

### 3. Pluggable Reference Lookup
- **Composite Pattern**: Multiple reference providers with priority ordering
- **Batch Operations**: Efficient bulk lookups
- **Context-Aware**: References can consider problem context
- **Extensible**: Easy to add new reference sources (APIs, databases, files)

### 4. Decomposition Templates
- **Pattern Matching**: Regex-based problem identification
- **Domain-Specific**: Templates tagged by applicable domains
- **Confidence Scoring**: Templates ranked by match confidence
- **Dynamic Substitution**: Templates adapt to specific problem details

## Usage Examples

### Basic Enhancement

```python
from compositional_prompting.enhancements import (
    EnhancementOrchestrator,
    EnhancedComposingPrompt,
    ExampleGraph,
    Example
)

# Create orchestrator with all enhancements
orchestrator = EnhancementOrchestrator()

# Configure example graph
graph = ExampleGraph()
graph.add_example(Example(
    id="ex1",
    content="For optimization, check KKT conditions",
    domain="mathematics"
))
orchestrator.with_example_graph(graph)

# Use enhanced fluid API
prompt = (
    EnhancedComposingPrompt(orchestrator=orchestrator)
    .problem_context("Maximize f(x) subject to g(x) ≤ 0")
    .cognitive_op(CognitiveOperation.ANALYZE)
    .with_smart_examples(n=3, strategy="diverse")
    .with_decomposition(domain="mathematics")
    .with_references(["KKT conditions"])
    .build_enhanced()
)
```

### Time-Bounded Operations

```python
from compositional_prompting.enhancements import time_bounded, with_timeout

# Use time-bounded context manager
with time_bounded(100) as timer:
    if timer.remaining_ms > 50:
        expensive_operation()
    else:
        quick_fallback()

    print(f"Elapsed: {timer.elapsed_ms}ms")

# Or use decorator
@with_timeout(timeout_ms=200)
def augment_prompt(prompt, timer=None):
    if timer and timer.remaining_ms > 100:
        return complex_augmentation(prompt)
    return simple_template(prompt)
```

### Graph-Based Example Retrieval

```python
# Create rich example graph
graph = ExampleGraph(similarity_threshold=0.6)

# Add examples (automatically creates edges)
for example in examples:
    graph.add_example(example)

# Detect communities
communities = graph.detect_communities()

# Find influential examples
hubs = graph.find_hubs(top_k=5)

# Find bridges between communities
bridges = graph.find_bridges()

# Retrieve with different strategies
similar = graph.retrieve_similar(query, k=5, strategy="similar")
diverse = graph.retrieve_similar(query, k=5, strategy="diverse")
```

### Pluggable Reference System

```python
from compositional_prompting.enhancements import (
    CompositeReferenceProvider,
    DictionaryReferenceProvider
)

# Create composite provider
provider = CompositeReferenceProvider()

# Add multiple backends with priorities
provider.add_provider(
    "mathematics",
    math_reference_api,
    priority=20
)
provider.add_provider(
    "fallback",
    local_dictionary,
    priority=10
)

# Lookup with automatic fallback
result = provider.lookup("eigenvalue")

# Batch lookup
results = provider.batch_lookup(["matrix", "vector", "tensor"])
```

## Design Principles

### 1. Clean Separation of Concerns
- Core library remains pure and deterministic
- Enhancements are entirely optional
- Each component is independently useful

### 2. Time-Bounded Everything
- All operations respect time constraints
- Graceful degradation when time expires
- Never blocks or hangs

### 3. Composable APIs
- Fluent interfaces that chain naturally
- Components work independently or together
- Easy to test and mock

### 4. Intelligence Without Critical Decisions
- Small LLM acts as coordinator, not decision maker
- Templates and patterns provide deterministic fallbacks
- System works even without LLM

## Performance Characteristics

- **Graph Operations**: O(1) retrieval with caching, O(n) for community detection
- **Augmentation**: 100-200ms typical, with caching for repeated queries
- **Reference Lookup**: Parallel queries to multiple providers
- **Template Matching**: Sub-millisecond pattern matching

## Testing

Comprehensive test suite covering:
- Time-bounded operations and graceful degradation
- Graph algorithms and retrieval strategies
- Augmentation caching and fallbacks
- Reference provider composition
- Template matching and application
- End-to-end integration scenarios
- Performance under load

Run tests:
```bash
pytest tests/test_enhancements.py -v
```

## Future Extensions

The enhancement layer is designed for extensibility:

- **Additional RAG Strategies**: Implement more sophisticated retrieval algorithms
- **Vector Embeddings**: Add semantic similarity with real embeddings
- **External Knowledge Bases**: Connect to Wikipedia, documentation, papers
- **Learning Components**: Add simple learning from feedback
- **Async Operations**: Support async/await patterns
- **Streaming Results**: Return results as they become available

## Philosophy

The enhancement layer embodies several key philosophies:

- **Unix Philosophy**: Each component does one thing well
- **Fail Gracefully**: Time bounds ensure responsiveness
- **Compose Don't Complicate**: Simple pieces that work together
- **Optional Enhancement**: Core system works without enhancements
- **Interpretable Operations**: Every enhancement is explainable

The result is a system that augments human and AI reasoning capabilities while maintaining transparency, determinism where needed, and beautiful APIs that are a joy to use.