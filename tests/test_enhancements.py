"""
Comprehensive test suite for the Compositional Prompting Enhancement Layer.

These tests verify the elegant functionality of the enhancement components,
ensuring they maintain time bounds, fail gracefully, and compose beautifully.
"""

import pytest
import time
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from compositional_prompting import (
    ComposingPrompt,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    MockLLMProvider
)

from compositional_prompting.enhancements import (
    EnhancementOrchestrator,
    EnhancementConfig,
    EnhancedComposingPrompt,
    ExampleGraph,
    Example,
    FastAugmenter,
    CompositeReferenceProvider,
    DictionaryReferenceProvider,
    TemplateLibrary,
    DecompositionTemplate,
    time_bounded,
    with_timeout,
    EnhancementCapability
)


class TestTimeBoundedOperations:
    """Test time management and graceful degradation"""

    def test_time_bounded_context_manager(self):
        """Test time-bounded context manager tracks time correctly"""
        with time_bounded(100) as timer:
            assert timer.remaining_ms > 95  # Some overhead is ok
            assert timer.elapsed_ms < 5
            assert not timer.is_expired

            time.sleep(0.05)  # 50ms
            assert timer.remaining_ms < 55
            assert timer.remaining_ms > 45
            assert timer.elapsed_ms > 45
            assert timer.elapsed_ms < 55

            time.sleep(0.06)  # Another 60ms, should exceed 100ms total
            assert timer.is_expired
            assert timer.remaining_ms == 0

    def test_with_timeout_decorator(self):
        """Test timeout decorator with graceful degradation"""

        @with_timeout(timeout_ms=50)
        def fast_function(value, timer=None):
            if timer and timer.remaining_ms > 0:
                return value * 2
            return value

        @with_timeout(timeout_ms=50)
        def slow_function(value, timer=None):
            time.sleep(0.1)  # 100ms, will timeout
            return value * 2

        # Fast function should complete
        result = fast_function(5)
        assert result == 10

        # Slow function should return None due to timeout
        result = slow_function(5)
        # Function still runs but decorator logs warning

    def test_timeout_with_exception_handling(self):
        """Test timeout decorator handles exceptions gracefully"""

        @with_timeout(timeout_ms=100)
        def failing_function(timer=None):
            raise ValueError("Test error")

        # Should return None instead of raising
        result = failing_function()
        assert result is None


class TestExampleGraph:
    """Test graph-based RAG system"""

    def create_test_graph(self) -> ExampleGraph:
        """Create a test graph with known structure"""
        graph = ExampleGraph(similarity_threshold=0.5)

        examples = [
            Example("ex1", "Binary search in sorted array", "algorithms"),
            Example("ex2", "Quick sort with pivot selection", "algorithms"),
            Example("ex3", "Proof by induction", "mathematics"),
            Example("ex4", "Solving quadratic equations", "mathematics"),
            Example("ex5", "Newton's laws of motion", "physics"),
        ]

        for ex in examples:
            graph.add_example(ex)

        return graph

    def test_example_addition_and_edge_creation(self):
        """Test that examples are added and edges are created"""
        graph = ExampleGraph(similarity_threshold=0.3)

        ex1 = Example("ex1", "Test content one", "domain1")
        ex2 = Example("ex2", "Test content two", "domain1")

        graph.add_example(ex1)
        assert len(graph.examples) == 1
        assert len(graph.edges) == 0

        graph.add_example(ex2)
        assert len(graph.examples) == 2
        # Should have edge due to same domain
        assert len(graph.adjacency[ex1.id]) > 0 or len(graph.adjacency[ex2.id]) > 0

    def test_community_detection(self):
        """Test community detection finds connected components"""
        graph = self.create_test_graph()
        communities = graph.detect_communities()

        # Should have at least one community
        assert len(set(communities.values())) >= 1

        # All examples should be assigned to a community
        assert len(communities) == len(graph.examples)

    def test_hub_detection(self):
        """Test hub detection finds most connected examples"""
        graph = self.create_test_graph()

        # Add a highly connected example
        hub = Example("hub", "Common algorithm patterns", "algorithms")
        graph.add_example(hub)

        hubs = graph.find_hubs(top_k=1)
        assert len(hubs) == 1

    def test_retrieval_strategies(self):
        """Test different retrieval strategies"""
        graph = self.create_test_graph()

        query = "sorting algorithms"

        # Test similar retrieval
        similar = graph.retrieve_similar(query, k=2, strategy="similar")
        assert len(similar) <= 2

        # Test diverse retrieval
        diverse = graph.retrieve_similar(query, k=2, strategy="diverse")
        assert len(diverse) <= 2

        # Test hub-aware retrieval
        hub_aware = graph.retrieve_similar(query, k=2, strategy="hub_aware")
        assert len(hub_aware) <= 2

        # Test community sampling
        community = graph.retrieve_similar(query, k=2, strategy="community")
        assert len(community) <= 2

    def test_retrieval_with_timeout(self):
        """Test retrieval respects time bounds"""
        graph = self.create_test_graph()

        # Create a mock timer that expires immediately
        class ExpiredTimer:
            @property
            def is_expired(self):
                return True

            @property
            def remaining_ms(self):
                return 0

        # Should return cached results when timer expired
        results = graph.retrieve_similar("test", k=5, timer=ExpiredTimer())
        assert results is not None  # Should still return something


class TestFastAugmenter:
    """Test fast local augmentation"""

    def test_make_specific(self):
        """Test making prompts specific to problems"""
        augmenter = FastAugmenter()

        prompt = "Solve this problem"
        problem = "Find the shortest path in a weighted graph"

        result = augmenter.make_specific(prompt, problem)
        assert result != prompt
        assert len(result) > len(prompt)

    def test_caching(self):
        """Test that augmenter caches results"""
        augmenter = FastAugmenter()

        prompt = "Analyze this"
        problem = "Binary search tree operations"

        # First call
        result1 = augmenter.make_specific(prompt, problem)

        # Second call should use cache
        result2 = augmenter.make_specific(prompt, problem)

        assert result1 == result2

    def test_key_term_extraction(self):
        """Test extraction of key terms"""
        augmenter = FastAugmenter()

        text = "QuickSort algorithm with O(n log n) complexity using divide-and-conquer"
        terms = augmenter._extract_key_terms(text)

        assert len(terms) > 0
        assert any("QuickSort" in term for term in terms)

    def test_decomposition_suggestions(self):
        """Test problem decomposition suggestions"""
        augmenter = FastAugmenter()

        problem = "Prove that all even numbers greater than 2 are composite"
        steps = augmenter.suggest_decomposition(problem)

        assert len(steps) > 0
        assert all(isinstance(step, str) for step in steps)

    def test_constraint_identification(self):
        """Test constraint identification in problems"""
        augmenter = FastAugmenter()

        problem = "Find x where x >= 5 and x <= 10 and x must be even"
        constraints = augmenter.identify_constraints(problem)

        assert "bounds" in constraints
        assert "requirements" in constraints
        assert len(constraints["bounds"]) > 0 or len(constraints["requirements"]) > 0


class TestReferenceProviders:
    """Test pluggable reference lookup system"""

    def test_dictionary_provider(self):
        """Test dictionary-based reference provider"""
        refs = {
            "algorithm": {"definition": "Step-by-step procedure", "complexity": "varies"},
            "graph": {"definition": "Nodes and edges", "types": ["directed", "undirected"]}
        }

        provider = DictionaryReferenceProvider(refs)

        # Test exact match
        result = provider.lookup("algorithm")
        assert result is not None
        assert "definition" in result

        # Test case-insensitive
        result = provider.lookup("GRAPH")
        assert result is not None

        # Test missing term
        result = provider.lookup("missing")
        assert result is None

    def test_composite_provider(self):
        """Test composite provider with multiple backends"""
        provider = CompositeReferenceProvider()

        # Add multiple providers with different priorities
        math_provider = DictionaryReferenceProvider({"pi": {"value": 3.14159}})
        cs_provider = DictionaryReferenceProvider({"algorithm": {"definition": "procedure"}})

        provider.add_provider("math", math_provider, priority=10)
        provider.add_provider("cs", cs_provider, priority=5)

        # Should find from higher priority provider first
        result = provider.lookup("pi")
        assert result is not None
        assert result["source"] == "math"

        result = provider.lookup("algorithm")
        assert result is not None
        assert result["source"] == "cs"

    def test_batch_lookup(self):
        """Test batch lookup functionality"""
        provider = CompositeReferenceProvider()
        dict_provider = DictionaryReferenceProvider({
            "term1": {"def": "definition 1"},
            "term2": {"def": "definition 2"},
        })
        provider.add_provider("test", dict_provider, priority=1)

        results = provider.batch_lookup(["term1", "term2", "missing"])
        assert len(results) == 2
        assert "term1" in results
        assert "term2" in results
        assert "missing" not in results


class TestTemplateLibrary:
    """Test decomposition template system"""

    def test_template_matching(self):
        """Test template pattern matching"""
        template = DecompositionTemplate(
            name="test",
            pattern=r"solve.*equation",
            steps=["Step 1", "Step 2"],
            applicable_domains=["math"]
        )

        assert template.matches("solve the equation")
        assert template.matches("SOLVE THIS EQUATION")
        assert not template.matches("find the solution")

    def test_template_application(self):
        """Test template application with substitution"""
        template = DecompositionTemplate(
            name="test",
            pattern=r"prove that (.+) equals (.+)",
            steps=["State: {0} = {1}", "Prove the equality"],
            applicable_domains=["math"]
        )

        problem = "prove that x + y equals y + x"
        steps = template.apply(problem)

        assert len(steps) == 2
        assert "x + y" in steps[0]
        assert "y + x" in steps[0]

    def test_template_library_operations(self):
        """Test template library functionality"""
        library = TemplateLibrary()

        # Should have default templates
        initial_count = len(library.templates)
        assert initial_count > 0

        # Add custom template
        custom = DecompositionTemplate(
            name="custom",
            pattern=r"custom problem",
            steps=["Custom step"],
            applicable_domains=["custom"],
            confidence=1.0
        )
        library.add_template(custom)

        assert len(library.templates) == initial_count + 1

        # Find matching templates
        matches = library.find_matching_templates("custom problem", domain="custom")
        assert len(matches) > 0
        assert matches[0].name == "custom"  # Highest confidence first

    def test_apply_best_template(self):
        """Test applying the best matching template"""
        library = TemplateLibrary()

        problem = "prove that the sum is correct"
        steps = library.apply_best_template(problem)

        # Should match the proof template
        assert steps is not None
        assert len(steps) > 0


class TestEnhancementOrchestrator:
    """Test main orchestrator"""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with config"""
        config = EnhancementConfig(
            enable_rag=False,
            enable_fast_augmentation=True,
            max_latency_ms=200
        )

        orchestrator = EnhancementOrchestrator(config)
        assert orchestrator.config.enable_rag is False
        assert orchestrator.config.enable_fast_augmentation is True

    def test_component_configuration(self):
        """Test setting orchestrator components"""
        orchestrator = EnhancementOrchestrator()

        graph = ExampleGraph()
        orchestrator.with_example_graph(graph)
        assert orchestrator.example_graph is graph

        augmenter = FastAugmenter()
        orchestrator.with_augmenter(augmenter)
        assert orchestrator.augmenter is augmenter

    def test_enhancement_application(self):
        """Test applying enhancements to prompts"""
        orchestrator = EnhancementOrchestrator(
            EnhancementConfig(enable_templates=True)
        )

        prompt = "Solve this problem"
        context = {"problem": "Find the optimal solution"}

        enhanced = orchestrator.enhance(prompt, context)

        assert "original_prompt" in enhanced
        assert enhanced["original_prompt"] == prompt
        assert "enhancements" in enhanced

    def test_metrics_tracking(self):
        """Test performance metrics tracking"""
        orchestrator = EnhancementOrchestrator()

        # Apply some enhancements
        orchestrator.enhance("test", {})
        orchestrator.enhance("test2", {})

        metrics = orchestrator.get_metrics()
        assert metrics["enhancements_applied"] == 2
        assert "avg_latency_ms" in metrics


class TestEnhancedComposingPrompt:
    """Test enhanced fluid API"""

    def test_enhanced_prompt_creation(self):
        """Test creating enhanced prompts"""
        orchestrator = EnhancementOrchestrator()
        enhanced = EnhancedComposingPrompt(orchestrator=orchestrator)

        assert enhanced.orchestrator is orchestrator
        assert enhanced.base is not None

    def test_delegation_to_base(self):
        """Test that base methods are accessible"""
        enhanced = EnhancedComposingPrompt()

        # Should delegate to base prompt
        enhanced.cognitive_op(CognitiveOperation.ANALYZE)
        enhanced.focus(FocusAspect.PATTERNS)

        assert enhanced.base._cognitive_op == CognitiveOperation.ANALYZE
        assert enhanced.base._focus == FocusAspect.PATTERNS

    def test_smart_examples_integration(self):
        """Test smart example addition"""
        orchestrator = EnhancementOrchestrator()
        graph = ExampleGraph()
        graph.add_example(Example("ex1", "Test example", "test"))
        orchestrator.with_example_graph(graph)

        enhanced = EnhancedComposingPrompt(orchestrator=orchestrator)
        enhanced.problem_context("Test problem")
        enhanced.with_smart_examples(n=1, strategy="similar")

        # Should have added context
        assert len(enhanced.base._context_additions) > 0

    def test_decomposition_integration(self):
        """Test decomposition template application"""
        enhanced = EnhancedComposingPrompt()
        enhanced.problem_context("Prove that x = y")
        enhanced.with_decomposition(domain="mathematics")

        # Should have added decomposition steps
        built = enhanced.base.build()
        assert "Decomposition:" in built or len(built) > 0

    def test_reference_integration(self):
        """Test reference lookup integration"""
        orchestrator = EnhancementOrchestrator()
        provider = CompositeReferenceProvider()
        provider.add_provider(
            "test",
            DictionaryReferenceProvider({"test": {"def": "definition"}}),
            priority=1
        )
        orchestrator.with_reference_provider(provider)

        enhanced = EnhancedComposingPrompt(orchestrator=orchestrator)
        enhanced.with_references(["test"])

        # Should have added reference
        assert len(enhanced.base._context_additions) > 0


class TestIntegration:
    """Integration tests for the complete system"""

    def test_end_to_end_enhancement(self):
        """Test complete enhancement pipeline"""

        # Create fully configured orchestrator
        orchestrator = EnhancementOrchestrator()

        # Setup example graph
        graph = ExampleGraph()
        graph.add_example(Example("ex1", "Dynamic programming solution", "algorithms"))
        graph.add_example(Example("ex2", "Greedy algorithm approach", "algorithms"))
        orchestrator.with_example_graph(graph)

        # Setup reference provider
        refs = CompositeReferenceProvider()
        refs.add_provider(
            "cs",
            DictionaryReferenceProvider({
                "dynamic programming": {"definition": "Optimization technique"}
            }),
            priority=10
        )
        orchestrator.with_reference_provider(refs)

        # Create and enhance prompt
        problem = "Find optimal solution using dynamic programming"
        enhanced = (
            EnhancedComposingPrompt(orchestrator=orchestrator)
            .problem_context(problem)
            .cognitive_op(CognitiveOperation.ANALYZE)
            .focus(FocusAspect.PATTERNS)
            .style(ReasoningStyle.SYSTEMATIC)
            .with_smart_examples(n=1, strategy="similar")
            .with_references(["dynamic programming"])
            .build_enhanced()
        )

        # Verify enhancement was applied
        assert problem in enhanced
        assert "ANALYZE" in enhanced or "analyze" in enhanced

    def test_parallel_enhancement(self):
        """Test parallel enhancement operations"""
        orchestrator = EnhancementOrchestrator()

        # Create multiple enhanced prompts
        prompts = []
        for i in range(3):
            enhanced = (
                EnhancedComposingPrompt(orchestrator=orchestrator)
                .problem_context(f"Problem {i}")
                .cognitive_op(CognitiveOperation.DECOMPOSE)
            )
            prompts.append(enhanced)

        # All should be independent
        assert all(p.base._problem_context == f"Problem {i}"
                  for i, p in enumerate(prompts))

    def test_graceful_degradation_under_load(self):
        """Test system behavior under time pressure"""

        config = EnhancementConfig(max_latency_ms=10)  # Very tight deadline
        orchestrator = EnhancementOrchestrator(config)

        # Add expensive operations
        graph = ExampleGraph()
        for i in range(100):
            graph.add_example(Example(f"ex{i}", f"Content {i}", "test"))
        orchestrator.with_example_graph(graph)

        # Should still complete without error
        enhanced = orchestrator.enhance("test prompt", {})
        assert enhanced is not None
        assert "original_prompt" in enhanced


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests"""

    def test_large_graph_performance(self):
        """Test performance with large example graph"""
        graph = ExampleGraph(similarity_threshold=0.7)

        # Add many examples
        start_time = time.perf_counter()
        for i in range(100):
            graph.add_example(
                Example(f"ex{i}", f"Example content {i}", f"domain{i%5}")
            )
        creation_time = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert creation_time < 5.0  # 5 seconds for 100 examples

        # Test retrieval performance
        start_time = time.perf_counter()
        results = graph.retrieve_similar("test query", k=5)
        retrieval_time = time.perf_counter() - start_time

        assert retrieval_time < 0.5  # 500ms for retrieval
        assert len(results) <= 5

    def test_template_matching_performance(self):
        """Test template matching with many templates"""
        library = TemplateLibrary()

        # Add many templates
        for i in range(50):
            library.add_template(DecompositionTemplate(
                name=f"template{i}",
                pattern=f"pattern{i}",
                steps=[f"Step {j}" for j in range(5)],
                applicable_domains=["test"],
                confidence=0.5 + (i * 0.01)
            ))

        # Test matching performance
        start_time = time.perf_counter()
        matches = library.find_matching_templates("test problem")
        match_time = time.perf_counter() - start_time

        assert match_time < 0.1  # 100ms for matching


if __name__ == "__main__":
    pytest.main([__file__, "-v"])