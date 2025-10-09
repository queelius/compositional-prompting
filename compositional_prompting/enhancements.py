"""
Compositional Prompting Enhancement Layer
A pure, composable enhancement system that augments the core library with optional intelligence.

Design Philosophy:
- Pure functions and immutable data structures where possible
- Lazy evaluation and streaming interfaces
- Pluggable backends with consistent interfaces
- Time-bounded operations with graceful degradation
- Clear separation between deterministic and probabilistic components
"""

from typing import List, Optional, Dict, Any, Protocol, Iterator, Tuple, Set, Callable, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import time
import logging
from contextlib import contextmanager
from functools import lru_cache, wraps
import hashlib
import json
from collections import defaultdict, deque
import heapq
import math

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ============================================================================
# Core Protocols and Abstractions
# ============================================================================

class EnhancementCapability(Enum):
    """Types of enhancement capabilities available"""
    RAG_EXAMPLES = "rag_examples"
    FAST_REFORMULATION = "fast_reformulation"
    REFERENCE_LOOKUP = "reference_lookup"
    DECOMPOSITION_TEMPLATES = "decomposition_templates"
    CONSTRAINT_DETECTION = "constraint_detection"
    CONCEPT_EXTRACTION = "concept_extraction"


class EnhancementProvider(Protocol):
    """Protocol for enhancement providers"""

    def capabilities(self) -> Set[EnhancementCapability]:
        """Return set of capabilities this provider supports"""
        ...

    def enhance(self, context: Dict[str, Any], capability: EnhancementCapability,
                timeout_ms: int = 200) -> Optional[Dict[str, Any]]:
        """Apply enhancement within time bound"""
        ...


# ============================================================================
# Time Management and Graceful Degradation
# ============================================================================

@contextmanager
def time_bounded(timeout_ms: int = 200):
    """Context manager for time-bounded operations"""
    start = time.perf_counter()
    deadline = start + (timeout_ms / 1000.0)

    class TimeContext:
        @property
        def remaining_ms(self) -> float:
            return max(0, (deadline - time.perf_counter()) * 1000)

        @property
        def elapsed_ms(self) -> float:
            return (time.perf_counter() - start) * 1000

        @property
        def is_expired(self) -> bool:
            return time.perf_counter() >= deadline

    yield TimeContext()


def with_timeout(timeout_ms: int = 200):
    """Decorator for time-bounded functions with graceful degradation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with time_bounded(timeout_ms) as timer:
                try:
                    # Pass timer as kwarg if function accepts it
                    import inspect
                    sig = inspect.signature(func)
                    if 'timer' in sig.parameters:
                        kwargs['timer'] = timer

                    result = func(*args, **kwargs)

                    if timer.is_expired:
                        logger.warning(f"{func.__name__} exceeded timeout of {timeout_ms}ms")

                    return result

                except Exception as e:
                    logger.error(f"{func.__name__} failed: {e}")
                    return None

        return wrapper
    return decorator


# ============================================================================
# Graph-Based RAG for Examples
# ============================================================================

@dataclass
class Example:
    """Represents an example with embeddings and metadata"""
    id: str
    content: str
    domain: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    quality_score: float = 0.5

    def __hash__(self):
        return hash(self.id)


@dataclass
class ExampleEdge:
    """Represents similarity between examples"""
    source: str
    target: str
    similarity: float
    edge_type: str = "semantic"  # semantic, structural, domain

    def __lt__(self, other):
        return self.similarity < other.similarity


class ExampleGraph:
    """
    Graph-based example storage with community detection and advanced retrieval.

    This implements a sophisticated RAG system where examples form a graph with
    edges based on various similarity metrics. Communities, hubs, and bridges
    provide intelligent retrieval patterns.
    """

    def __init__(self, similarity_threshold: float = 0.7):
        self.examples: Dict[str, Example] = {}
        self.edges: List[ExampleEdge] = []
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.similarity_threshold = similarity_threshold
        self.communities: Optional[Dict[str, int]] = None
        self._embedder: Optional[Callable] = None

    def set_embedder(self, embedder: Callable[[str], List[float]]) -> 'ExampleGraph':
        """Set the embedding function for computing similarities"""
        self._embedder = embedder
        return self

    def add_example(self, example_or_id, content=None, metadata=None) -> 'ExampleGraph':
        """Add an example and compute edges to existing examples

        Args:
            example_or_id: Either an Example object, or an ID string
            content: Content string (if example_or_id is ID string)
            metadata: Metadata dict (if example_or_id is ID string)
        """
        if isinstance(example_or_id, Example):
            example = example_or_id
        else:
            # Create Example from individual parameters
            example = Example(
                id=example_or_id,
                content=content or "",
                domain=metadata.get('domain', 'general') if metadata else 'general',
                metadata=metadata or {}
            )

        self.examples[example.id] = example

        # Compute similarities and add edges
        for other_id, other in self.examples.items():
            if other_id != example.id:
                similarity = self._compute_similarity(example, other)
                if similarity >= self.similarity_threshold:
                    edge = ExampleEdge(example.id, other_id, similarity)
                    self.edges.append(edge)
                    self.adjacency[example.id].add(other_id)
                    self.adjacency[other_id].add(example.id)

        # Invalidate communities cache
        self.communities = None
        return self

    def _compute_similarity(self, ex1: Example, ex2: Example) -> float:
        """Compute similarity between two examples"""
        # Domain similarity
        domain_sim = 1.0 if ex1.domain == ex2.domain else 0.3

        # Embedding similarity if available
        if ex1.embedding and ex2.embedding:
            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(ex1.embedding, ex2.embedding))
            norm1 = math.sqrt(sum(a * a for a in ex1.embedding))
            norm2 = math.sqrt(sum(b * b for b in ex2.embedding))
            embedding_sim = dot_product / (norm1 * norm2 + 1e-8)
        else:
            # Fallback to simple text similarity
            embedding_sim = self._text_similarity(ex1.content, ex2.content)

        # Weighted combination
        return 0.3 * domain_sim + 0.7 * embedding_sim

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for text"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / (len(union) + 1e-8)

    @lru_cache(maxsize=128)
    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using Louvain-like algorithm"""
        if not self.examples:
            return {}

        # Simple community detection via connected components
        visited = set()
        communities = {}
        community_id = 0

        for example_id in self.examples:
            if example_id not in visited:
                # BFS to find connected component
                queue = deque([example_id])
                visited.add(example_id)

                while queue:
                    current = queue.popleft()
                    communities[current] = community_id

                    for neighbor in self.adjacency[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                community_id += 1

        self.communities = communities
        return communities

    def find_hubs(self, top_k: int = 5) -> List[Example]:
        """Find hub examples with most connections"""
        degree_map = {
            ex_id: len(self.adjacency[ex_id])
            for ex_id in self.examples
        }

        top_examples = heapq.nlargest(
            top_k,
            degree_map.items(),
            key=lambda x: x[1]
        )

        return [self.examples[ex_id] for ex_id, _ in top_examples]

    def find_bridges(self) -> List[Example]:
        """Find bridge examples connecting different communities"""
        communities = self.detect_communities()
        bridges = []

        for edge in self.edges:
            comm1 = communities.get(edge.source)
            comm2 = communities.get(edge.target)

            if comm1 != comm2:
                bridges.append(self.examples[edge.source])
                bridges.append(self.examples[edge.target])

        # Remove duplicates while preserving order
        seen = set()
        unique_bridges = []
        for ex in bridges:
            if ex.id not in seen:
                seen.add(ex.id)
                unique_bridges.append(ex)

        return unique_bridges

    @with_timeout(timeout_ms=100)
    def retrieve_similar(self, query: str, k: int = 5,
                        strategy: str = "diverse", timer=None) -> List[Example]:
        """
        Retrieve similar examples using various strategies.

        Strategies:
        - 'similar': Pure similarity-based retrieval
        - 'diverse': Mix of similar and diverse examples
        - 'hub_aware': Prefer hub examples
        - 'community': Sample from different communities
        """

        if not self.examples:
            return []

        # Quick timeout check
        if timer and timer.is_expired:
            # Return cached best examples
            return list(self.examples.values())[:k]

        try:
            if strategy == "similar":
                return self._retrieve_by_similarity(query, k)
            elif strategy == "diverse":
                return self._retrieve_diverse(query, k)
            elif strategy == "hub_aware":
                return self._retrieve_hub_aware(query, k)
            elif strategy == "community":
                return self._retrieve_community_sample(query, k)
            else:
                return self._retrieve_by_similarity(query, k)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Fallback to simple list
            return list(self.examples.values())[:k]

    def _retrieve_by_similarity(self, query: str, k: int) -> List[Example]:
        """Pure similarity-based retrieval"""
        # Compute query similarity to all examples
        scores = []
        for ex_id, example in self.examples.items():
            similarity = self._text_similarity(query, example.content)
            scores.append((similarity, example))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scores[:k]]

    def _retrieve_diverse(self, query: str, k: int) -> List[Example]:
        """Retrieve mix of similar and diverse examples"""
        similar = self._retrieve_by_similarity(query, k // 2)

        # Get examples from different communities
        communities = self.detect_communities()
        community_samples = defaultdict(list)

        for ex_id, comm_id in communities.items():
            if self.examples[ex_id] not in similar:
                community_samples[comm_id].append(self.examples[ex_id])

        diverse = []
        for comm_examples in community_samples.values():
            if comm_examples:
                diverse.append(comm_examples[0])
            if len(diverse) >= k - len(similar):
                break

        return similar + diverse[:k - len(similar)]

    def _retrieve_hub_aware(self, query: str, k: int) -> List[Example]:
        """Prefer hub examples in retrieval"""
        hubs = self.find_hubs(k * 2)

        # Score hubs by similarity to query
        scored_hubs = []
        for hub in hubs:
            similarity = self._text_similarity(query, hub.content)
            scored_hubs.append((similarity, hub))

        scored_hubs.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored_hubs[:k]]

    def _retrieve_community_sample(self, query: str, k: int) -> List[Example]:
        """Sample from different communities"""
        communities = self.detect_communities()
        community_groups = defaultdict(list)

        for ex_id, comm_id in communities.items():
            community_groups[comm_id].append(self.examples[ex_id])

        # Sample evenly from communities
        samples = []
        per_community = max(1, k // len(community_groups)) if community_groups else k

        for comm_examples in community_groups.values():
            # Sort by similarity within community
            comm_scored = []
            for ex in comm_examples:
                similarity = self._text_similarity(query, ex.content)
                comm_scored.append((similarity, ex))

            comm_scored.sort(key=lambda x: x[0], reverse=True)
            samples.extend([ex for _, ex in comm_scored[:per_community]])

        return samples[:k]


# ============================================================================
# Fast Local LLM Augmentation
# ============================================================================

class FastAugmenter:
    """
    Fast, local LLM augmentation for context preparation.
    Uses small models for quick reformulation without solving.
    """

    def __init__(self, model_provider: Optional[Any] = None,
                 max_latency_ms: int = 100):
        self.model_provider = model_provider
        self.max_latency_ms = max_latency_ms
        self.cache = {}  # Simple cache for repeated queries

    @with_timeout(timeout_ms=100)
    def make_specific(self, prompt: str, problem: str, timer=None) -> str:
        """
        Make a generic prompt specific to the problem without solving it.
        This is a fast operation that adds context, not intelligence.
        """

        # Check cache first
        cache_key = hashlib.md5(f"{prompt}:{problem}".encode()).hexdigest()
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Quick timeout check
        if timer and timer.remaining_ms < 10:
            return prompt  # Return original if no time

        # Extract key terms from problem
        key_terms = self._extract_key_terms(problem)

        # Simple template-based augmentation (no LLM needed for speed)
        augmented = f"{prompt}\n\nSpecifically for this problem involving {', '.join(key_terms)}:"

        # If we have time and model, do smart augmentation
        if timer and timer.remaining_ms > 50 and self.model_provider:
            try:
                quick_prompt = f"Rephrase concisely for problem about {key_terms[0]}: {prompt[:100]}"
                result = self.model_provider.generate(
                    quick_prompt,
                    max_tokens=50,
                    temperature=0.3
                )
                if result:
                    augmented = result
            except:
                pass  # Fallback to template

        self.cache[cache_key] = augmented
        return augmented

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text (simple version)"""
        # Simple heuristic: find capitalized words and technical terms
        import re

        # Find potential key terms
        words = text.split()
        key_terms = []

        # Look for: capitalized words, words with special chars, long words
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)
            if (len(cleaned) > 7 or
                word[0].isupper() or
                any(c in word for c in ['_', '-', '/', '\\'])):
                key_terms.append(cleaned)

        return list(set(key_terms))[:5]  # Return top 5 unique terms

    @with_timeout(timeout_ms=150)
    def suggest_decomposition(self, problem: str, timer=None) -> List[str]:
        """Suggest problem decomposition steps quickly"""

        # Pattern-based decomposition (fast, no LLM)
        steps = []

        # Look for natural break points
        if "and" in problem.lower():
            steps.append("Break down compound conditions")

        if any(word in problem.lower() for word in ["all", "every", "each"]):
            steps.append("Consider individual cases")

        if any(char in problem for char in "∀∃∈⊂⊆"):
            steps.append("Decompose mathematical quantifiers")

        if "?" in problem or "find" in problem.lower():
            steps.append("Identify what needs to be determined")

        if not steps:
            # Generic decomposition
            steps = [
                "Identify key components",
                "Determine constraints",
                "Find solution approach"
            ]

        return steps

    @with_timeout(timeout_ms=100)
    def identify_constraints(self, problem: str, timer=None) -> Dict[str, Any]:
        """Quickly identify constraints in the problem"""

        constraints = {
            "bounds": [],
            "conditions": [],
            "requirements": []
        }

        # Pattern matching for common constraints
        import re

        # Numerical bounds
        bounds_patterns = [
            r'(?:between|from)\s+(\d+)\s+(?:to|and)\s+(\d+)',
            r'(?:at most|maximum|max|≤|<=)\s+(\d+)',
            r'(?:at least|minimum|min|≥|>=)\s+(\d+)',
            r'(?:exactly|equal to|=)\s+(\d+)'
        ]

        for pattern in bounds_patterns:
            matches = re.findall(pattern, problem, re.IGNORECASE)
            if matches:
                constraints["bounds"].extend(matches)

        # Logical conditions
        if "if" in problem.lower():
            constraints["conditions"].append("conditional logic")

        if "must" in problem.lower() or "should" in problem.lower():
            constraints["requirements"].append("mandatory requirement")

        return constraints


# ============================================================================
# Pluggable Reference Lookup System
# ============================================================================

class ReferenceProvider(Protocol):
    """Protocol for reference providers"""

    def lookup(self, term: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Lookup a term and return reference information"""
        ...

    def batch_lookup(self, terms: List[str], context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Batch lookup multiple terms"""
        ...


class CompositeReferenceProvider:
    """
    Composite reference provider that queries multiple backends.
    Implements a chain of responsibility with fallbacks.
    """

    def __init__(self):
        self.providers: List[Tuple[str, ReferenceProvider, int]] = []  # (name, provider, priority)

    def add_provider(self, name: str, provider: ReferenceProvider, priority: int = 0) -> 'CompositeReferenceProvider':
        """Add a reference provider with priority (higher = checked first)"""
        self.providers.append((name, provider, priority))
        self.providers.sort(key=lambda x: x[2], reverse=True)
        return self

    @with_timeout(timeout_ms=200)
    def lookup(self, term: str, context: Optional[str] = None, timer=None) -> Optional[Dict[str, Any]]:
        """Lookup term across all providers"""

        for name, provider, _ in self.providers:
            if timer and timer.is_expired:
                break

            try:
                result = provider.lookup(term, context)
                if result:
                    result['source'] = name
                    return result
            except Exception as e:
                logger.warning(f"Provider {name} failed: {e}")
                continue

        return None

    def batch_lookup(self, terms: List[str], context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Batch lookup with parallel execution"""
        results = {}

        for term in terms:
            result = self.lookup(term, context)
            if result:
                results[term] = result

        return results


class DictionaryReferenceProvider:
    """Simple dictionary-based reference provider"""

    def __init__(self, references: Dict[str, Dict[str, Any]]):
        self.references = references

    def lookup(self, term: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Lookup term in dictionary"""
        # Try exact match
        if term in self.references:
            return self.references[term]

        # Try case-insensitive
        term_lower = term.lower()
        for key, value in self.references.items():
            if key.lower() == term_lower:
                return value

        return None

    def batch_lookup(self, terms: List[str], context: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Batch lookup"""
        return {term: ref for term in terms if (ref := self.lookup(term, context))}


# ============================================================================
# Decomposition and Constraint Templates
# ============================================================================

@dataclass
class DecompositionTemplate:
    """Template for problem decomposition"""
    name: str
    pattern: str  # Pattern to match
    steps: List[str]  # Decomposition steps
    applicable_domains: List[str] = field(default_factory=list)
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, problem: str) -> bool:
        """Check if template matches the problem"""
        import re
        return bool(re.search(self.pattern, problem, re.IGNORECASE))

    def apply(self, problem: str) -> List[str]:
        """Apply template to generate decomposition steps"""
        import re

        # Extract matched groups
        match = re.search(self.pattern, problem, re.IGNORECASE)
        if not match:
            return self.steps

        # Substitute placeholders in steps
        applied_steps = []
        for step in self.steps:
            try:
                # Simple substitution of {0}, {1}, etc. with matched groups
                formatted = step
                for i, group in enumerate(match.groups()):
                    formatted = formatted.replace(f"{{{i}}}", str(group))
                applied_steps.append(formatted)
            except:
                applied_steps.append(step)

        return applied_steps


class TemplateLibrary:
    """Library of decomposition and constraint templates"""

    def __init__(self):
        self.templates: List[DecompositionTemplate] = []
        self._initialize_default_templates()

    def _initialize_default_templates(self):
        """Initialize with common templates"""

        # Mathematical optimization template
        self.add_template(DecompositionTemplate(
            name="optimization",
            pattern=r"(minimize|maximize|optimize)\s+(.+?)\s+subject to",
            steps=[
                "Define objective function: {1}",
                "Identify all constraints",
                "Check constraint compatibility",
                "Apply optimization technique",
                "Verify solution satisfies constraints"
            ],
            applicable_domains=["mathematics", "optimization"]
        ))

        # Proof template
        self.add_template(DecompositionTemplate(
            name="proof",
            pattern=r"prove\s+(?:that\s+)?(.+)",
            steps=[
                "State what needs to be proven: {0}",
                "Identify given information",
                "Choose proof strategy",
                "Execute proof steps",
                "Conclude with QED"
            ],
            applicable_domains=["mathematics", "logic"]
        ))

        # Algorithm design template
        self.add_template(DecompositionTemplate(
            name="algorithm",
            pattern=r"(design|implement|create)\s+(?:an?\s+)?algorithm",
            steps=[
                "Define input/output specification",
                "Identify algorithmic approach",
                "Design data structures",
                "Implement core logic",
                "Analyze complexity"
            ],
            applicable_domains=["computer science", "algorithms"]
        ))

        # System design template
        self.add_template(DecompositionTemplate(
            name="system_design",
            pattern=r"design\s+(?:a\s+)?system",
            steps=[
                "Gather requirements",
                "Define system boundaries",
                "Identify components",
                "Design interfaces",
                "Consider scalability and reliability"
            ],
            applicable_domains=["software engineering", "systems"]
        ))

    def add_template(self, template: DecompositionTemplate) -> 'TemplateLibrary':
        """Add a template to the library"""
        self.templates.append(template)
        return self

    def find_matching_templates(self, problem: str,
                               domain: Optional[str] = None) -> List[DecompositionTemplate]:
        """Find all templates matching the problem"""
        matches = []

        for template in self.templates:
            # Check domain match if specified
            if domain and template.applicable_domains:
                if domain not in template.applicable_domains:
                    continue

            # Check pattern match
            if template.matches(problem):
                matches.append(template)

        # Sort by confidence
        matches.sort(key=lambda t: t.confidence, reverse=True)
        return matches

    def apply_best_template(self, problem: str,
                           domain: Optional[str] = None) -> Optional[List[str]]:
        """Apply the best matching template"""
        templates = self.find_matching_templates(problem, domain)
        if templates:
            return templates[0].apply(problem)
        return None

    def search_templates(self, predicate: Callable[[DecompositionTemplate], bool]) -> List[DecompositionTemplate]:
        """Search templates using a predicate function"""
        return [template for template in self.templates if predicate(template)]

    def find_best_template(self, problem: str, domain: Optional[str] = None) -> Optional[DecompositionTemplate]:
        """Find the best matching template for a problem"""
        templates = self.find_matching_templates(problem, domain)
        return templates[0] if templates else None


# ============================================================================
# Main Enhancement Orchestrator
# ============================================================================

@dataclass
class EnhancementConfig:
    """Configuration for enhancement layer"""
    enable_rag: bool = True
    enable_fast_augmentation: bool = True
    enable_reference_lookup: bool = True
    enable_templates: bool = True
    max_latency_ms: int = 500
    parallel_operations: bool = True


class EnhancementOrchestrator:
    """
    Main orchestrator for the enhancement layer.
    Coordinates all enhancement components with clean, composable API.
    """

    def __init__(self, config: Optional[EnhancementConfig] = None, default_timeout: Optional[float] = None):
        if default_timeout is not None:
            # Convert seconds to milliseconds and create config
            config_kwargs = {'max_latency_ms': int(default_timeout * 1000)}
            if config is not None:
                # Merge with existing config
                config_dict = vars(config).copy()
                config_dict.update(config_kwargs)
                self.config = EnhancementConfig(**config_dict)
            else:
                self.config = EnhancementConfig(**config_kwargs)
        else:
            self.config = config or EnhancementConfig()

        # Initialize components
        self.example_graph = ExampleGraph()
        self.augmenter = FastAugmenter()
        self.reference_provider = CompositeReferenceProvider()
        self.template_library = TemplateLibrary()

        # Metrics tracking
        self.metrics = {
            "enhancements_applied": 0,
            "total_latency_ms": 0,
            "cache_hits": 0,
            "timeouts": 0
        }

    def with_example_graph(self, graph: ExampleGraph) -> 'EnhancementOrchestrator':
        """Set the example graph"""
        self.example_graph = graph
        return self

    def with_augmenter(self, augmenter: FastAugmenter) -> 'EnhancementOrchestrator':
        """Set the fast augmenter"""
        self.augmenter = augmenter
        return self

    def with_reference_provider(self, provider: CompositeReferenceProvider) -> 'EnhancementOrchestrator':
        """Set the reference provider"""
        self.reference_provider = provider
        return self

    def with_template_library(self, library: TemplateLibrary) -> 'EnhancementOrchestrator':
        """Set the template library"""
        self.template_library = library
        return self

    @with_timeout(timeout_ms=500)
    def enhance(self, prompt: str, context: Dict[str, Any], timer=None) -> Dict[str, Any]:
        """
        Apply all configured enhancements to the prompt.

        Returns enhanced context with all augmentations.
        """

        enhanced = {
            "original_prompt": prompt,
            "enhancements": {}
        }

        start_time = time.perf_counter()

        # Apply RAG for examples
        if self.config.enable_rag and timer and timer.remaining_ms > 100:
            examples = self.example_graph.retrieve_similar(
                prompt,
                k=3,
                strategy="diverse"
            )
            if examples:
                enhanced["enhancements"]["examples"] = [ex.content for ex in examples]

        # Apply fast augmentation
        if self.config.enable_fast_augmentation and timer and timer.remaining_ms > 50:
            problem = context.get("problem", "")
            if problem:
                augmented = self.augmenter.make_specific(prompt, problem)
                enhanced["enhancements"]["augmented_prompt"] = augmented

        # Apply reference lookup
        if self.config.enable_reference_lookup and timer and timer.remaining_ms > 50:
            # Extract terms to lookup
            terms = self.augmenter._extract_key_terms(prompt)
            if terms:
                references = self.reference_provider.batch_lookup(terms[:3])
                if references:
                    enhanced["enhancements"]["references"] = references

        # Apply decomposition templates
        if self.config.enable_templates:
            problem = context.get("problem", prompt)
            decomposition = self.template_library.apply_best_template(problem)
            if decomposition:
                enhanced["enhancements"]["decomposition"] = decomposition

        # Update metrics
        self.metrics["enhancements_applied"] += 1
        self.metrics["total_latency_ms"] += (time.perf_counter() - start_time) * 1000

        return enhanced

    def apply_enhancements(self, prompt: str, capabilities: Optional[set] = None,
                          config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply specified enhancements to the prompt.

        Args:
            prompt: The prompt to enhance
            capabilities: Set of capabilities to apply (for backward compatibility)
            config: Configuration parameters (for backward compatibility)
        """
        context = config or {}
        return self.enhance(prompt, context)

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "avg_latency_ms": self.metrics["total_latency_ms"] / max(1, self.metrics["enhancements_applied"])
        }


# ============================================================================
# Fluid API Integration
# ============================================================================

class EnhancedComposingPrompt:
    """
    Enhanced prompt composer that integrates with the enhancement layer.
    Provides a fluid API that seamlessly combines core and enhanced features.
    """

    def __init__(self, base_prompt=None, orchestrator: Optional[EnhancementOrchestrator] = None):
        # Import the base class
        from . import ComposingPrompt

        self.base = base_prompt or ComposingPrompt()
        self.orchestrator = orchestrator or EnhancementOrchestrator()
        self.enhancement_context = {}

    def __getattr__(self, name):
        """Delegate to base prompt for all standard methods"""
        attr = getattr(self.base, name)
        # If it's a method, wrap it to return self for chaining
        if callable(attr):
            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                # If the result is the base prompt, return self instead for chaining
                if result is self.base:
                    return self
                return result
            return wrapper
        return attr

    def with_orchestrator(self, orchestrator: EnhancementOrchestrator) -> 'EnhancedComposingPrompt':
        """Set the enhancement orchestrator"""
        self.orchestrator = orchestrator
        return self

    def enhance(self) -> 'EnhancedComposingPrompt':
        """Apply enhancements to the current prompt"""
        current_prompt = self.base.build()
        enhanced_context = self.orchestrator.enhance(
            current_prompt,
            self.enhancement_context
        )

        # Apply enhancements back to base prompt
        if "augmented_prompt" in enhanced_context.get("enhancements", {}):
            self.base.add_context(enhanced_context["enhancements"]["augmented_prompt"])

        if "examples" in enhanced_context.get("enhancements", {}):
            for example in enhanced_context["enhancements"]["examples"]:
                self.base.add_context(f"Example: {example}")

        if "decomposition" in enhanced_context.get("enhancements", {}):
            steps = enhanced_context["enhancements"]["decomposition"]
            self.base.add_context("Decomposition:\n" + "\n".join(f"- {s}" for s in steps))

        return self

    def with_smart_examples(self, n: int = 3, strategy: str = "diverse") -> 'EnhancedComposingPrompt':
        """Add examples using graph-based RAG"""
        examples = self.orchestrator.example_graph.retrieve_similar(
            self.base._problem_context or "",
            k=n,
            strategy=strategy
        )

        for example in examples:
            self.base.add_context(f"Example: {example.content}")

        return self

    def with_decomposition(self, domain: Optional[str] = None) -> 'EnhancedComposingPrompt':
        """Apply decomposition template"""
        problem = self.base._problem_context or self.base._base_prompt or ""
        decomposition = self.orchestrator.template_library.apply_best_template(problem, domain)

        if decomposition:
            self.base.add_context("Decomposition:\n" + "\n".join(f"- {s}" for s in decomposition))

        return self

    def with_references(self, terms: Optional[List[str]] = None) -> 'EnhancedComposingPrompt':
        """Add reference lookups for key terms"""
        if not terms:
            # Auto-extract terms
            text = self.base._problem_context or self.base._base_prompt or ""
            terms = self.orchestrator.augmenter._extract_key_terms(text)

        if terms:
            references = self.orchestrator.reference_provider.batch_lookup(terms)
            for term, ref in references.items():
                self.base.add_context(f"Reference [{term}]: {ref}")

        return self

    def build_enhanced(self) -> str:
        """Build with all enhancements applied"""
        self.enhance()
        return self.base.build()


# ============================================================================
# Example Usage and Demo
# ============================================================================

def create_demo_orchestrator() -> EnhancementOrchestrator:
    """Create a demo orchestrator with sample data"""

    orchestrator = EnhancementOrchestrator()

    # Add sample examples to graph
    example_graph = ExampleGraph()
    example_graph.add_example(Example(
        id="ex1",
        content="To solve |x| + |y| ≤ 5, consider the four quadrants separately",
        domain="mathematics"
    ))
    example_graph.add_example(Example(
        id="ex2",
        content="For combinatorial problems, count systematically using inclusion-exclusion",
        domain="mathematics"
    ))
    example_graph.add_example(Example(
        id="ex3",
        content="Dynamic programming: break into subproblems with optimal substructure",
        domain="algorithms"
    ))

    # Setup reference provider
    ref_provider = CompositeReferenceProvider()
    ref_provider.add_provider(
        "math_dict",
        DictionaryReferenceProvider({
            "absolute value": {"definition": "Distance from zero", "notation": "|x|"},
            "combinatorics": {"definition": "Study of counting", "techniques": ["permutations", "combinations"]},
            "dynamic programming": {"definition": "Optimization over recursive subproblems", "complexity": "O(n^2) typical"}
        }),
        priority=10
    )

    orchestrator.with_example_graph(example_graph)
    orchestrator.with_reference_provider(ref_provider)

    return orchestrator


if __name__ == "__main__":
    # Demo the enhancement layer

    print("=== Compositional Prompting Enhancement Layer Demo ===\n")

    # Create orchestrator
    orchestrator = create_demo_orchestrator()

    # Test time-bounded operations
    print("1. Time-bounded operation test:")
    with time_bounded(100) as timer:
        print(f"  Starting with {timer.remaining_ms:.1f}ms remaining")
        time.sleep(0.05)
        print(f"  After 50ms: {timer.remaining_ms:.1f}ms remaining")
        print(f"  Is expired: {timer.is_expired}")

    # Test example graph
    print("\n2. Example graph operations:")
    graph = orchestrator.example_graph
    print(f"  Total examples: {len(graph.examples)}")
    print(f"  Communities detected: {len(set(graph.detect_communities().values()))}")

    similar = graph.retrieve_similar("absolute value inequality", k=2)
    print(f"  Retrieved {len(similar)} similar examples")
    for ex in similar:
        print(f"    - {ex.content[:50]}...")

    # Test fast augmentation
    print("\n3. Fast augmentation:")
    augmenter = orchestrator.augmenter
    prompt = "Solve this step by step"
    problem = "Find all integer solutions to |x| + |y| ≤ 5"
    augmented = augmenter.make_specific(prompt, problem)
    print(f"  Original: {prompt}")
    print(f"  Augmented: {augmented}")

    # Test decomposition templates
    print("\n4. Decomposition templates:")
    library = orchestrator.template_library
    problem = "Prove that the sum of two even numbers is even"
    steps = library.apply_best_template(problem)
    if steps:
        print(f"  Problem: {problem}")
        print("  Decomposition:")
        for step in steps:
            print(f"    - {step}")

    # Test full enhancement
    print("\n5. Full enhancement:")
    context = {"problem": "Design an algorithm to find the shortest path in a graph"}
    enhanced = orchestrator.enhance("Solve this problem", context)

    print("  Enhancements applied:")
    for key, value in enhanced.get("enhancements", {}).items():
        print(f"    - {key}: {type(value).__name__}")

    # Show metrics
    print("\n6. Performance metrics:")
    metrics = orchestrator.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n=== Demo Complete ===")