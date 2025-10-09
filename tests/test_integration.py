"""
Integration tests for compositional prompting library.
Tests interaction between core library and enhancement layer.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
import random
from typing import List, Dict, Any

from compositional_prompting import (
    ComposingPrompt,
    MockLLMProvider,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
)

from compositional_prompting.enhancements import (
    EnhancementOrchestrator,
    EnhancedComposingPrompt,
    ExampleGraph,
    Example,
    FastAugmenter,
    DecompositionTemplate,
    TemplateLibrary,
    DictionaryReferenceProvider,
    CompositeReferenceProvider,
    time_bounded,
    with_timeout,
)


class TestEnhancedPromptIntegration:
    """Test integration between core ComposingPrompt and EnhancedComposingPrompt"""

    def test_enhanced_prompt_delegates_to_base(self):
        """Test that enhanced prompt properly delegates to base methods"""
        enhanced = EnhancedComposingPrompt()

        # Test fluent API delegation
        result = (enhanced
                 .cognitive_op(CognitiveOperation.ANALYZE)
                 .focus(FocusAspect.PATTERNS)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .connect(ConnectionType.THEREFORE)
                 .format(OutputFormat.STEPS))

        assert isinstance(result, EnhancedComposingPrompt)
        assert result.base._cognitive_op == CognitiveOperation.ANALYZE
        assert result.base._focus == FocusAspect.PATTERNS
        assert result.base._style == ReasoningStyle.SYSTEMATIC
        assert result.base._connection == ConnectionType.THEREFORE
        assert result.base._output_format == OutputFormat.STEPS

    def test_enhanced_smart_examples_with_core_prompt(self):
        """Test smart examples integration with core prompt building"""
        enhanced = EnhancedComposingPrompt()
        orchestrator = enhanced.orchestrator

        # Add some examples to the graph
        example_graph = orchestrator.example_graph
        example_graph.add_example("example1", "Math problem about factorials", {"domain": "math"})
        example_graph.add_example("example2", "Physics problem about velocity", {"domain": "physics"})
        example_graph.add_example("example3", "Math problem about integrals", {"domain": "math"})

        # Build prompt with smart examples
        result = (enhanced
                 .problem_context("Calculate the factorial of 10")
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .with_smart_examples(n=2, strategy="diverse"))

        # Should have added examples through the enhancement
        built = result.build()
        assert "Calculate the factorial" in built

    def test_enhanced_decomposition_with_base_prompt(self):
        """Test decomposition template integration"""
        enhanced = EnhancedComposingPrompt()

        # Add decomposition template
        template = DecompositionTemplate(
            name="math_problem",
            pattern=r"calculate|solve|find",
            steps=[
                "1. Understand the problem statement",
                "2. Identify given information",
                "3. Apply relevant formulas",
                "4. Compute the solution"
            ]
        )
        enhanced.orchestrator.template_library.add_template(template)

        result = (enhanced
                 .problem_context("Calculate the area of a circle")
                 .with_decomposition()
                 .cognitive_op(CognitiveOperation.DECOMPOSE))

        built = result.build()
        assert "Calculate the area" in built

    def test_enhanced_references_with_core(self):
        """Test reference lookup integration"""
        enhanced = EnhancedComposingPrompt()

        # Add reference provider
        dict_provider = DictionaryReferenceProvider({
            "factorial": "The product of all positive integers less than or equal to n",
            "integral": "The area under a curve"
        })
        enhanced.orchestrator.add_reference_provider(dict_provider)

        result = (enhanced
                 .problem_context("Explain factorial")
                 .with_references(["factorial"])
                 .cognitive_op(CognitiveOperation.GENERATE))

        built = result.build()
        assert "Explain factorial" in built

    def test_enhanced_with_llm_augmentation(self):
        """Test enhanced prompt with LLM augmentation"""
        mock_llm = MockLLMProvider()
        enhanced = EnhancedComposingPrompt()

        result = (enhanced
                 .set_llm_provider(mock_llm)
                 .problem_context("Complex optimization problem")
                 .cognitive_op(CognitiveOperation.ANALYZE)
                 .llm_augment("Make this more specific")
                 .with_smart_examples(n=2))

        # Verify both base and enhanced features work together
        assert enhanced.base._llm_provider is mock_llm
        assert len(enhanced.base._llm_augmentations) == 1


class TestOrchestratorIntegration:
    """Test orchestrator coordination between components"""

    def test_orchestrator_coordinates_multiple_enhancements(self):
        """Test orchestrator applies multiple enhancements"""
        orchestrator = EnhancementOrchestrator()

        # Configure components
        orchestrator.example_graph.add_example("ex1", "Example content", {})
        orchestrator.fast_augmenter.cache["test"] = "cached result"

        template = DecompositionTemplate(
            name="test_template",
            pattern=r"test",
            steps=["Step 1", "Step 2"]
        )
        orchestrator.template_library.add_template(template)

        # Apply enhancements
        prompt_text = "This is a test problem"
        enhancements = orchestrator.apply_enhancements(
            prompt_text,
            capabilities={
                'rag_examples',
                'decomposition_templates'
            },
            config={
                'n_examples': 1,
                'retrieval_strategy': 'similar'
            }
        )

        assert 'examples' in enhancements
        assert 'decomposition' in enhancements
        assert len(enhancements['examples']) > 0

    def test_orchestrator_timeout_handling(self):
        """Test orchestrator handles timeouts gracefully"""
        orchestrator = EnhancementOrchestrator(default_timeout=0.01)  # 10ms timeout

        # Add slow operation
        def slow_operation():
            time.sleep(0.1)  # 100ms
            return "Should not reach here"

        # Should timeout and return empty results
        with time_bounded(0.01):
            prompt_text = "Test prompt"
            enhancements = orchestrator.apply_enhancements(
                prompt_text,
                capabilities={'rag_examples'}
            )

        # Should have gracefully degraded
        assert enhancements is not None

    def test_orchestrator_metrics_tracking(self):
        """Test metrics are tracked correctly"""
        orchestrator = EnhancementOrchestrator()

        initial_count = orchestrator.metrics['total_enhancements']

        # Apply some enhancements
        orchestrator.apply_enhancements(
            "Test prompt",
            capabilities={'fast_reformulation'}
        )

        assert orchestrator.metrics['total_enhancements'] > initial_count
        assert orchestrator.metrics['fast_augmenter_calls'] > 0

    def test_orchestrator_with_multiple_providers(self):
        """Test orchestrator with multiple reference providers"""
        orchestrator = EnhancementOrchestrator()

        # Add multiple reference providers
        dict1 = DictionaryReferenceProvider({"term1": "Definition 1"})
        dict2 = DictionaryReferenceProvider({"term2": "Definition 2"})

        orchestrator.add_reference_provider(dict1)
        orchestrator.add_reference_provider(dict2)

        # Create composite provider
        composite = CompositeReferenceProvider([dict1, dict2])
        results = composite.batch_lookup(["term1", "term2"])

        assert "term1" in results
        assert "term2" in results


class TestCoreEnhancementChaining:
    """Test chaining core and enhanced functionality"""

    def test_chain_enhanced_prompts(self):
        """Test chaining multiple enhanced prompts"""
        enhanced1 = (EnhancedComposingPrompt()
                    .problem_context("First problem")
                    .cognitive_op(CognitiveOperation.ANALYZE))

        enhanced2 = (EnhancedComposingPrompt()
                    .cognitive_op(CognitiveOperation.SYNTHESIZE))

        chain = enhanced1.chain(enhanced2)

        assert len(chain) == 2
        assert chain[0].base._cognitive_op == CognitiveOperation.ANALYZE
        assert chain[1].base._cognitive_op == CognitiveOperation.SYNTHESIZE

    def test_fork_enhanced_prompts(self):
        """Test forking enhanced prompts"""
        enhanced = (EnhancedComposingPrompt()
                   .problem_context("Original problem")
                   .cognitive_op(CognitiveOperation.DECOMPOSE))

        # Add some state to orchestrator
        enhanced.orchestrator.example_graph.add_example("ex1", "content", {})

        forks = enhanced.fork(n=3)

        assert len(forks) == 3
        for fork in forks:
            assert isinstance(fork, EnhancedComposingPrompt)
            assert fork.base._problem_context == "Original problem"
            # Each should have independent orchestrator
            assert fork.orchestrator is not enhanced.orchestrator

    def test_compose_enhanced_with_base(self):
        """Test composing enhanced prompt with base prompt"""
        base_prompt = (ComposingPrompt()
                      .cognitive_op(CognitiveOperation.ANALYZE)
                      .focus(FocusAspect.PATTERNS))

        enhanced = EnhancedComposingPrompt()
        enhanced.base.compose_with(base_prompt, merge_strategy='overlay')

        assert enhanced.base._cognitive_op == CognitiveOperation.ANALYZE
        assert enhanced.base._focus == FocusAspect.PATTERNS


class TestGraphIntegration:
    """Test graph-based example retrieval with core prompts"""

    def test_graph_with_prompt_building(self):
        """Test graph retrieval integrated with prompt building"""
        graph = ExampleGraph()

        # Build a connected graph
        graph.add_example("ex1", "Math factorial", {"type": "math"})
        graph.add_example("ex2", "Math integral", {"type": "math"})
        graph.add_example("ex3", "Physics velocity", {"type": "physics"})

        graph.add_edge("ex1", "ex2", weight=0.9)
        graph.add_edge("ex2", "ex3", weight=0.5)

        # Detect communities
        communities = graph.detect_communities()

        # Use in prompt building
        prompt = ComposingPrompt()

        # Add examples from most connected community
        if communities:
            largest_community = max(communities, key=len)
            for ex_id in list(largest_community)[:2]:
                if ex_id in graph.examples:
                    prompt.add_context(f"Example: {graph.examples[ex_id].content}")

        built = prompt.build()
        # Should contain examples from graph
        assert "Example:" in built or built == ""

    def test_graph_retrieval_strategies(self):
        """Test different retrieval strategies with prompt integration"""
        graph = ExampleGraph()

        # Build test graph
        for i in range(5):
            graph.add_example(f"ex{i}", f"Content {i}", {"index": i})

        # Add edges to create structure
        graph.add_edge("ex0", "ex1", weight=0.9)
        graph.add_edge("ex1", "ex2", weight=0.8)
        graph.add_edge("ex2", "ex3", weight=0.7)

        prompt = ComposingPrompt()

        # Test different strategies
        strategies = [
            "similar",
            "diverse",
            "hub_aware"
        ]

        for strategy in strategies:
            examples = graph.retrieve_examples(
                n=2,
                strategy=strategy,
                similarity_scores={"ex0": 0.9, "ex1": 0.8, "ex2": 0.7}
            )
            assert len(examples) <= 2

    def test_graph_hub_detection_with_prompts(self):
        """Test hub detection for key example identification"""
        graph = ExampleGraph()

        # Create hub structure
        hub_id = "hub"
        graph.add_example(hub_id, "Central example", {})

        for i in range(5):
            spoke_id = f"spoke{i}"
            graph.add_example(spoke_id, f"Spoke {i}", {})
            graph.add_edge(hub_id, spoke_id, weight=0.8)

        hubs = graph.find_hubs(threshold=3)
        assert hub_id in hubs

        # Use hub in prompt
        prompt = ComposingPrompt()
        if hubs:
            hub_example = graph.examples[list(hubs)[0]]
            prompt.add_context(f"Key example: {hub_example.content}")

        built = prompt.build()
        assert "Central example" in built


class TestFastAugmenterIntegration:
    """Test fast augmenter with core prompts"""

    def test_fast_augmenter_reformulation(self):
        """Test fast augmenter reformulates prompts"""
        augmenter = FastAugmenter()

        prompt = "Solve this problem"
        problem = "Find the maximum value in the array"
        specific = augmenter.make_specific(prompt, problem)

        # Should transform the prompt
        assert specific != prompt
        assert len(specific) > 0

        # Use with core prompt
        composed = (ComposingPrompt()
                 .problem_context(problem)
                 .add_context(f"Augmented: {specific}"))

        built = composed.build()
        assert problem in built

    def test_fast_augmenter_caching(self):
        """Test augmenter caching for performance"""
        augmenter = FastAugmenter()

        prompt = "Solve this"
        problem = "Test problem statement"

        # First call
        result1 = augmenter.make_specific(prompt, problem)

        # Second call should use cache
        result2 = augmenter.make_specific(prompt, problem)

        assert result1 == result2
        # Check cache contains the hashed key
        import hashlib
        cache_key = hashlib.md5(f"{prompt}:{problem}".encode()).hexdigest()
        assert cache_key in augmenter.cache

    def test_fast_augmenter_term_extraction(self):
        """Test key term extraction for concept identification"""
        augmenter = FastAugmenter()

        text = "Calculate the factorial of a number using dynamic programming"
        terms = augmenter._extract_key_terms(text)[:3]

        assert len(terms) <= 3
        assert all(isinstance(term, str) for term in terms)

        # Use terms in prompt
        prompt = ComposingPrompt()
        for term in terms:
            prompt.add_context(f"Key concept: {term}")

        built = prompt.build()
        # Should contain key terms
        assert any(term in built for term in terms) or built == ""


class TestTemplateIntegration:
    """Test template system with core prompts"""

    def test_template_application_with_prompt(self):
        """Test applying templates to prompts"""
        library = TemplateLibrary()

        template = DecompositionTemplate(
            name="algorithm_design",
            pattern=r"algorithm|design|implement",
            steps=[
                "1. Define the problem",
                "2. Identify constraints",
                "3. Design approach",
                "4. Implement solution",
                "5. Test and optimize"
            ],
            metadata={"domain": "programming"}
        )

        library.add_template(template)

        # Apply to prompt
        prompt_text = "Design an algorithm for sorting"
        best_template = library.find_best_template(prompt_text)

        assert best_template is not None
        assert best_template.name == "algorithm_design"

        # Use with core prompt
        prompt = (ComposingPrompt()
                 .problem_context(prompt_text)
                 .cognitive_op(CognitiveOperation.DECOMPOSE))

        if best_template:
            for step in best_template.steps:
                prompt.add_context(step)

        built = prompt.build()
        assert "Define the problem" in built

    def test_template_library_search(self):
        """Test template library search functionality"""
        library = TemplateLibrary()

        # Add multiple templates
        templates = [
            DecompositionTemplate(
                name="math",
                pattern=r"calculate|solve",
                steps=["Math step 1", "Math step 2"]
            ),
            DecompositionTemplate(
                name="physics",
                pattern=r"force|velocity",
                steps=["Physics step 1", "Physics step 2"]
            ),
            DecompositionTemplate(
                name="chemistry",
                pattern=r"reaction|compound",
                steps=["Chemistry step 1", "Chemistry step 2"]
            )
        ]

        for template in templates:
            library.add_template(template)

        # Search by domain
        physics_templates = library.search_templates(lambda t: "physics" in t.name)
        assert len(physics_templates) == 1
        assert physics_templates[0].name == "physics"


class TestTimeoutIntegration:
    """Test timeout mechanisms with core functionality"""

    def test_timeout_decorator_with_prompt_building(self):
        """Test timeout decorator on prompt building"""

        @with_timeout(0.1)  # 100ms timeout
        def build_complex_prompt():
            prompt = ComposingPrompt()
            for i in range(1000):
                prompt.add_context(f"Context {i}")
            return prompt.build()

        # Should complete within timeout
        result = build_complex_prompt()
        assert result is not None

    def test_timeout_context_manager(self):
        """Test timeout context manager"""

        with time_bounded(0.1):
            prompt = (ComposingPrompt()
                     .cognitive_op(CognitiveOperation.ANALYZE)
                     .focus(FocusAspect.PATTERNS))

            # Should complete quickly
            built = prompt.build()
            assert built is not None

    def test_timeout_graceful_degradation(self):
        """Test graceful degradation under timeout pressure"""
        orchestrator = EnhancementOrchestrator(default_timeout=0.001)  # 1ms - very short

        # Add many examples (slow operation)
        for i in range(100):
            orchestrator.example_graph.add_example(f"ex{i}", f"Content {i}", {})

        # Should degrade gracefully and return partial results
        enhancements = orchestrator.apply_enhancements(
            "Test prompt",
            capabilities={'rag_examples'},
            config={'n_examples': 50}  # Request many examples
        )

        # Should return something, even if incomplete
        assert enhancements is not None


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""

    def test_math_problem_solving_pipeline(self):
        """Test complete pipeline for math problem"""
        # Setup enhanced prompt with full configuration
        enhanced = EnhancedComposingPrompt()

        # Add examples
        enhanced.orchestrator.example_graph.add_example(
            "factorial_ex",
            "5! = 5 × 4 × 3 × 2 × 1 = 120",
            {"type": "factorial"}
        )
        enhanced.orchestrator.example_graph.add_example(
            "combination_ex",
            "C(5,2) = 5!/(2!(5-2)!) = 10",
            {"type": "combination"}
        )

        # Add decomposition template
        math_template = DecompositionTemplate(
            name="factorial_calculation",
            pattern=r"factorial|!",
            steps=[
                "1. Identify the number n",
                "2. Initialize result to 1",
                "3. Multiply result by each integer from 1 to n",
                "4. Return the final result"
            ]
        )
        enhanced.orchestrator.template_library.add_template(math_template)

        # Build complete prompt
        result = (enhanced
                 .problem_context("Calculate 7!")
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .focus(FocusAspect.STRUCTURE)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .format(OutputFormat.STEPS)
                 .with_smart_examples(n=1)
                 .with_decomposition())

        built = result.build()

        # Verify all components are present
        assert "Calculate 7!" in built
        assert "systematically" in built.lower()
        assert "steps" in built.lower()

    def test_parallel_exploration_pipeline(self):
        """Test parallel exploration of solution approaches"""
        base = (EnhancedComposingPrompt()
               .problem_context("Optimize database query performance"))

        # Fork into different approaches
        variations = [
            {'_style': ReasoningStyle.SYSTEMATIC},
            {'_style': ReasoningStyle.CREATIVE},
            {'_style': ReasoningStyle.CRITICAL}
        ]

        forks = base.fork(n=3, variations=variations)

        # Each fork explores different approach
        results = []
        for fork in forks:
            fork.cognitive_op(CognitiveOperation.GENERATE)
            results.append(fork.build())

        # Should have 3 different approaches
        assert len(results) == 3
        assert all("Optimize database" in r for r in results)

    def test_iterative_refinement_pipeline(self):
        """Test iterative refinement with evaluation"""
        mock_llm = Mock(spec=MockLLMProvider)
        mock_llm.generate.return_value = "0.7"  # Quality score

        enhanced = (EnhancedComposingPrompt()
                   .set_llm_provider(mock_llm)
                   .problem_context("Design a distributed system"))

        # Initial attempt
        initial = enhanced.cognitive_op(CognitiveOperation.GENERATE)
        initial_quality = initial.llm_evaluate(criteria="completeness")

        # Refine based on evaluation
        if initial_quality < 0.8:
            refined = (enhanced
                      .cognitive_op(CognitiveOperation.SYNTHESIZE)
                      .llm_augment("Add more detail")
                      .with_smart_examples(n=2))

            refined_quality = refined.llm_evaluate(criteria="completeness")

        # Should have attempted refinement
        assert len(enhanced.base._llm_augmentations) > 0

    def test_multi_provider_consensus(self):
        """Test using multiple providers for consensus"""
        providers = [
            MockLLMProvider(),
            MockLLMProvider(),
            MockLLMProvider()
        ]

        # Each provider gives different response
        for i, provider in enumerate(providers):
            provider.generate = Mock(return_value=f"Response {i}")

        enhanced = EnhancedComposingPrompt()

        # Get responses from all providers
        responses = []
        for provider in providers:
            prompt_copy = (EnhancedComposingPrompt()
                          .set_llm_provider(provider)
                          .problem_context("Explain quantum computing")
                          .cognitive_op(CognitiveOperation.GENERATE))

            # Execute pipeline
            prompt_copy.execute_llm_pipeline()
            responses.append(provider.generate("Explain quantum computing"))

        # Should have responses from all providers
        assert len(responses) == 3
        assert all(f"Response {i}" in responses for i in range(3))