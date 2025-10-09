"""
Advanced test suite for compositional prompting core library.
Tests advanced composition methods, parallel execution, and action vectors.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import concurrent.futures
import copy
import random

from compositional_prompting import (
    ComposingPrompt,
    MockLLMProvider,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
    MetaStrategy,
    ConfidenceLevel,
    ReasoningDepth,
    create_provider_factory
)


class TestAdvancedComposition:
    """Test advanced composition methods"""

    def test_compose_with_overlay_strategy(self):
        """Test compose_with using overlay strategy"""
        prompt1 = (ComposingPrompt()
                  .cognitive_op(CognitiveOperation.DECOMPOSE)
                  .focus(FocusAspect.STRUCTURE)
                  .style(ReasoningStyle.SYSTEMATIC))

        prompt2 = (ComposingPrompt()
                  .cognitive_op(CognitiveOperation.ANALYZE)
                  .style(ReasoningStyle.CREATIVE)
                  .format(OutputFormat.STEPS))

        result = prompt1.compose_with(prompt2, merge_strategy='overlay')

        # prompt2 values should override prompt1
        assert result._cognitive_op == CognitiveOperation.ANALYZE
        assert result._style == ReasoningStyle.CREATIVE
        assert result._output_format == OutputFormat.STEPS
        # prompt1 value not overridden should remain
        assert result._focus == FocusAspect.STRUCTURE

    def test_compose_with_underlay_strategy(self):
        """Test compose_with using underlay strategy"""
        prompt1 = (ComposingPrompt()
                  .cognitive_op(CognitiveOperation.DECOMPOSE)
                  .focus(FocusAspect.STRUCTURE))

        prompt2 = (ComposingPrompt()
                  .cognitive_op(CognitiveOperation.ANALYZE)
                  .focus(FocusAspect.PATTERNS)
                  .style(ReasoningStyle.CREATIVE))

        result = prompt1.compose_with(prompt2, merge_strategy='underlay')

        # prompt1 values should be kept
        assert result._cognitive_op == CognitiveOperation.DECOMPOSE
        assert result._focus == FocusAspect.STRUCTURE
        # prompt2 should fill gaps
        assert result._style == ReasoningStyle.CREATIVE

    def test_compose_with_merge_strategy(self):
        """Test compose_with using merge strategy"""
        prompt1 = ComposingPrompt()
        prompt1.add_context("Context 1")
        prompt1.llm_augment("Augment 1")

        prompt2 = ComposingPrompt()
        prompt2.add_context("Context 2")
        prompt2.llm_augment("Augment 2")

        result = prompt1.compose_with(prompt2, merge_strategy='merge')

        # Should merge contexts and augmentations
        assert len(result._context_additions) == 2
        assert "Context 1" in result._context_additions
        assert "Context 2" in result._context_additions
        assert len(result._llm_augmentations) == 2

    def test_chain_method(self):
        """Test chaining multiple prompts"""
        prompt1 = ComposingPrompt().problem_context("First problem")
        prompt2 = ComposingPrompt().cognitive_op(CognitiveOperation.ANALYZE)
        prompt3 = ComposingPrompt().focus(FocusAspect.PATTERNS)

        chain = prompt1.chain(prompt2, prompt3)

        assert len(chain) == 3
        assert chain[0] is prompt1

        # Check that connection types are set
        assert prompt2._connection == ConnectionType.BUILDING_ON
        assert prompt3._connection == ConnectionType.BUILDING_ON

        # Check context accumulation
        assert "First problem" in prompt2._problem_context
        assert "[Previous context continues]" in prompt2._problem_context

    def test_chain_preserves_existing_connections(self):
        """Test that chain preserves manually set connections"""
        prompt1 = ComposingPrompt()
        prompt2 = ComposingPrompt().connect(ConnectionType.HOWEVER)

        chain = prompt1.chain(prompt2)

        # Should preserve the manually set connection
        assert prompt2._connection == ConnectionType.HOWEVER

    def test_fork_basic(self):
        """Test basic forking functionality"""
        original = (ComposingPrompt()
                   .cognitive_op(CognitiveOperation.ANALYZE)
                   .focus(FocusAspect.PATTERNS)
                   .problem_context("Original problem"))

        forks = original.fork(n=3)

        assert len(forks) == 3

        for fork in forks:
            # Each fork should be a deep copy
            assert fork is not original
            assert fork._cognitive_op == CognitiveOperation.ANALYZE
            assert fork._focus == FocusAspect.PATTERNS
            assert fork._problem_context == "Original problem"

            # Modifying fork shouldn't affect original
            fork._cognitive_op = CognitiveOperation.GENERATE
            assert original._cognitive_op == CognitiveOperation.ANALYZE

    def test_fork_with_variations(self):
        """Test forking with variations"""
        original = ComposingPrompt().cognitive_op(CognitiveOperation.ANALYZE)

        variations = [
            {'_style': ReasoningStyle.SYSTEMATIC},
            {'_style': ReasoningStyle.CREATIVE},
            {'_style': ReasoningStyle.CRITICAL}
        ]

        forks = original.fork(n=3, variations=variations)

        assert forks[0]._style == ReasoningStyle.SYSTEMATIC
        assert forks[1]._style == ReasoningStyle.CREATIVE
        assert forks[2]._style == ReasoningStyle.CRITICAL

        # All should retain original cognitive_op
        for fork in forks:
            assert fork._cognitive_op == CognitiveOperation.ANALYZE

    def test_fork_with_partial_variations(self):
        """Test forking with fewer variations than forks"""
        original = ComposingPrompt()

        variations = [
            {'_cognitive_op': CognitiveOperation.ANALYZE}
        ]

        forks = original.fork(n=3, variations=variations)

        assert forks[0]._cognitive_op == CognitiveOperation.ANALYZE
        # Other forks should not have the variation applied
        assert forks[1]._cognitive_op is None
        assert forks[2]._cognitive_op is None


class TestLLMEvaluation:
    """Test LLM evaluation methods"""

    def test_llm_evaluate_basic(self):
        """Test basic LLM evaluation"""
        mock_provider = Mock(spec=MockLLMProvider)
        mock_provider.generate.return_value = "0.85"

        prompt = (ComposingPrompt()
                 .set_llm_provider(mock_provider)
                 .cognitive_op(CognitiveOperation.ANALYZE))

        score = prompt.llm_evaluate(criteria="clarity")

        assert score == 0.85
        mock_provider.generate.assert_called_once()
        call_args = mock_provider.generate.call_args[0][0]
        assert "clarity" in call_args
        assert "Score (0-1):" in call_args

    def test_llm_evaluate_bounds_checking(self):
        """Test that evaluation scores are bounded between 0 and 1"""
        mock_provider = Mock(spec=MockLLMProvider)

        prompt = ComposingPrompt().set_llm_provider(mock_provider)

        # Test upper bound
        mock_provider.generate.return_value = "1.5"
        assert prompt.llm_evaluate() == 1.0

        # Test lower bound
        mock_provider.generate.return_value = "-0.5"
        assert prompt.llm_evaluate() == 0.0

    def test_llm_evaluate_fallback(self):
        """Test evaluation fallback when LLM fails"""
        mock_provider = Mock(spec=MockLLMProvider)
        mock_provider.generate.side_effect = Exception("API error")

        prompt = ComposingPrompt().set_llm_provider(mock_provider)
        score = prompt.llm_evaluate()

        assert score == 0.5  # Default fallback

    def test_llm_evaluate_with_custom_provider(self):
        """Test evaluation with custom provider"""
        default_provider = Mock(spec=MockLLMProvider)
        custom_provider = Mock(spec=MockLLMProvider)
        custom_provider.generate.return_value = "0.75"

        prompt = ComposingPrompt().set_llm_provider(default_provider)
        score = prompt.llm_evaluate(provider=custom_provider)

        assert score == 0.75
        custom_provider.generate.assert_called_once()
        default_provider.generate.assert_not_called()


class TestWeightedSampling:
    """Test weighted sampling functionality"""

    @patch('random.choice')
    def test_sample_weighted_uniform(self, mock_choice):
        """Test uniform sampling without weights"""
        prompt = ComposingPrompt()
        prompt.sample_weighted()

        # Should call random.choice for each component
        assert mock_choice.call_count == 5  # 5 components

    def test_sample_weighted_with_weights(self):
        """Test weighted sampling with provided weights"""
        weights = {
            'cognitive_op': {
                CognitiveOperation.ANALYZE: 10.0,
                CognitiveOperation.DECOMPOSE: 1.0
            }
        }

        # Run multiple samples to test distribution
        results = []
        for _ in range(100):
            prompt = ComposingPrompt()
            prompt.sample_weighted(weights)
            results.append(prompt._cognitive_op)

        # ANALYZE should appear much more often than DECOMPOSE
        analyze_count = results.count(CognitiveOperation.ANALYZE)
        decompose_count = results.count(CognitiveOperation.DECOMPOSE)

        # With 10:1 weight ratio, ANALYZE should appear significantly more
        # Allow for some statistical variation
        assert analyze_count > decompose_count

    def test_sample_weighted_partial_weights(self):
        """Test weighted sampling with weights for some components only"""
        weights = {
            'focus': {
                FocusAspect.PATTERNS: 5.0,
                FocusAspect.STRUCTURE: 1.0
            }
        }

        prompt = ComposingPrompt()
        prompt.sample_weighted(weights)

        # Focus should be weighted, others should be set
        assert prompt._focus is not None
        assert prompt._cognitive_op is not None
        assert prompt._style is not None

    def test_sample_action_class_method(self):
        """Test the sample_action class method"""
        weights = {
            'cognitive_op': {
                CognitiveOperation.ANALYZE: 2.0
            }
        }

        prompt = ComposingPrompt.sample_action(weights=weights)

        assert isinstance(prompt, ComposingPrompt)
        assert prompt._cognitive_op is not None
        assert prompt._focus is not None

    def test_sample_action_with_provider(self):
        """Test sample_action with custom provider"""
        custom_provider = MockLLMProvider()

        prompt = ComposingPrompt.sample_action(provider=custom_provider)

        assert prompt._llm_provider is custom_provider


class TestParallelExecution:
    """Test parallel LLM execution"""

    def test_execute_parallel_llm_calls(self):
        """Test parallel execution of LLM calls"""
        mock_provider1 = Mock(spec=MockLLMProvider)
        mock_provider1.generate.return_value = "Result 1"

        mock_provider2 = Mock(spec=MockLLMProvider)
        mock_provider2.generate.return_value = "Result 2"

        calls = [
            {'provider': mock_provider1, 'instruction': "Test 1"},
            {'provider': mock_provider2, 'instruction': "Test 2"}
        ]

        prompt = ComposingPrompt()
        results = prompt._execute_parallel_llm_calls(calls, max_workers=2)

        assert len(results) == 2
        assert "Result 1" in results
        assert "Result 2" in results

    def test_execute_llm_pipeline_parallel_examples(self):
        """Test pipeline execution with parallel example generation"""
        mock_provider = Mock(spec=MockLLMProvider)
        mock_provider.generate.return_value = "Generated example"

        prompt = (ComposingPrompt()
                 .set_llm_provider(mock_provider)
                 .llm_add_examples(n=3, parallel=True))

        prompt.execute_llm_pipeline(max_workers=2)

        # Should have made 3 parallel calls
        assert mock_provider.generate.call_count == 3
        assert len(prompt._examples) == 3

    def test_execute_llm_pipeline_sequential_augmentations(self):
        """Test pipeline execution with sequential augmentations"""
        mock_provider = Mock(spec=MockLLMProvider)
        mock_provider.generate.return_value = "Augmented content"

        prompt = (ComposingPrompt()
                 .set_llm_provider(mock_provider)
                 .llm_augment("First augmentation")
                 .llm_augment("Second augmentation"))

        prompt.execute_llm_pipeline()

        # Should have made 2 sequential calls
        assert mock_provider.generate.call_count == 2
        assert len(prompt._context_additions) == 2

    def test_execute_llm_pipeline_with_coherence_check(self):
        """Test pipeline execution with coherence check"""
        mock_provider = Mock(spec=MockLLMProvider)
        mock_provider.generate.return_value = "Coherence feedback"

        prompt = (ComposingPrompt()
                 .set_llm_provider(mock_provider)
                 .llm_coherence_check())

        prompt.execute_llm_pipeline()

        # Should have made coherence check call
        assert mock_provider.generate.called
        call_args = mock_provider.generate.call_args[0][0]
        assert "coherence" in call_args.lower()

    def test_execute_llm_pipeline_empty(self):
        """Test pipeline execution with no operations"""
        prompt = ComposingPrompt()
        result = prompt.execute_llm_pipeline()

        assert result is prompt  # Should return self
        assert len(prompt._examples) == 0
        assert len(prompt._context_additions) == 0


class TestActionVectors:
    """Test action vector functionality"""

    def test_get_action_vector_full(self):
        """Test action vector with all components set"""
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.ANALYZE)
                 .focus(FocusAspect.PATTERNS)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .connect(ConnectionType.THEREFORE)
                 .format(OutputFormat.STEPS)
                 .llm_augment("test")
                 .add_context("context"))

        vector = prompt.get_action_vector()

        assert vector['omega'] == 'analyze'
        assert vector['phi'] == 'patterns'
        assert vector['sigma'] == 'systematic'
        assert vector['kappa'] == 'therefore'
        assert vector['tau'] == 'steps'
        assert vector['llm_augmentations'] == 1
        assert vector['coherence_checks'] is False
        assert vector['context_additions'] == 1
        assert vector['has_parallel_ops'] is False

    def test_get_action_vector_partial(self):
        """Test action vector with partial components"""
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .style(ReasoningStyle.CREATIVE))

        vector = prompt.get_action_vector()

        assert vector['omega'] == 'decompose'
        assert vector['phi'] is None
        assert vector['sigma'] == 'creative'
        assert vector['kappa'] is None
        assert vector['tau'] is None

    def test_get_action_vector_with_parallel_ops(self):
        """Test action vector detects parallel operations"""
        prompt = ComposingPrompt().llm_add_examples(n=2, parallel=True)

        vector = prompt.get_action_vector()

        assert vector['has_parallel_ops'] is True

    def test_get_action_vector_empty(self):
        """Test action vector for empty prompt"""
        prompt = ComposingPrompt()
        vector = prompt.get_action_vector()

        assert vector['omega'] is None
        assert vector['phi'] is None
        assert vector['sigma'] is None
        assert vector['kappa'] is None
        assert vector['tau'] is None
        assert vector['llm_augmentations'] == 0
        assert vector['coherence_checks'] is False


class TestBuildMethods:
    """Test prompt building methods"""

    def test_build_compositional_core_full(self):
        """Test building core with all components"""
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .focus(FocusAspect.STRUCTURE)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .connect(ConnectionType.THEREFORE)
                 .format(OutputFormat.STEPS))

        core = prompt._build_compositional_core()

        assert "break this problem down" in core
        assert "structural relationships" in core
        assert "systematically" in core
        assert "Therefore," in core
        assert "clear steps:" in core

    def test_build_compositional_core_partial(self):
        """Test building core with partial components"""
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.ANALYZE)
                 .format(OutputFormat.LIST))

        core = prompt._build_compositional_core()

        assert "analyze this problem" in core
        assert "structured list:" in core

    def test_build_full_prompt(self):
        """Test building complete prompt"""
        prompt = (ComposingPrompt()
                 .base_prompt("Base instructions")
                 .problem_context("Solve this problem")
                 .cognitive_op(CognitiveOperation.GENERATE)
                 .focus(FocusAspect.SOLUTION)
                 .add_context("Additional context")
                 .llm_augment("Make it better"))

        result = prompt.build()

        assert "Base instructions" in result
        assert "Problem: Solve this problem" in result
        assert "generate a solution" in result
        assert "Additional context" in result
        assert "[LLM Augmentations to apply:" in result
        assert "Make it better" in result

    def test_build_with_examples(self):
        """Test building prompt with examples"""
        prompt = ComposingPrompt()
        prompt._examples = ["Example 1", "Example 2"]

        result = prompt.build()

        assert "Relevant examples:" in result
        assert "Example 1" in result
        assert "Example 2" in result

    def test_build_with_coherence_check(self):
        """Test building prompt with coherence check flag"""
        prompt = ComposingPrompt().llm_coherence_check()

        result = prompt.build()

        assert "[Apply coherence checking]" in result

    def test_build_empty_prompt(self):
        """Test building empty prompt"""
        prompt = ComposingPrompt()
        result = prompt.build()

        # Should return something, even if minimal
        assert result == ""


class TestProviderFactory:
    """Test provider factory function"""

    def test_create_provider_factory_mock(self):
        """Test factory creates mock provider"""
        factory = create_provider_factory()
        provider = factory("mock")

        assert isinstance(provider, MockLLMProvider)

    @patch('compositional_prompting.OpenAIProvider')
    def test_create_provider_factory_openai(self, mock_openai_class):
        """Test factory creates OpenAI provider"""
        mock_instance = Mock()
        mock_openai_class.return_value = mock_instance

        factory = create_provider_factory()
        provider = factory("openai", api_key="test-key", model="gpt-4")

        mock_openai_class.assert_called_once_with(api_key="test-key", model="gpt-4")
        assert provider is mock_instance

    @patch('compositional_prompting.AnthropicProvider')
    def test_create_provider_factory_anthropic(self, mock_anthropic_class):
        """Test factory creates Anthropic provider"""
        mock_instance = Mock()
        mock_anthropic_class.return_value = mock_instance

        factory = create_provider_factory()
        provider = factory("anthropic", api_key="test-key")

        mock_anthropic_class.assert_called_once_with(api_key="test-key")
        assert provider is mock_instance

    @patch('compositional_prompting.OllamaProvider')
    def test_create_provider_factory_ollama(self, mock_ollama_class):
        """Test factory creates Ollama provider"""
        mock_instance = Mock()
        mock_ollama_class.return_value = mock_instance

        factory = create_provider_factory()
        provider = factory("ollama", model="llama2")

        mock_ollama_class.assert_called_once_with(model="llama2")
        assert provider is mock_instance

    def test_create_provider_factory_unknown(self):
        """Test factory raises error for unknown provider"""
        factory = create_provider_factory()

        with pytest.raises(ValueError) as exc_info:
            factory("unknown_provider")

        assert "Unknown provider type" in str(exc_info.value)

    def test_create_provider_factory_case_insensitive(self):
        """Test factory is case insensitive"""
        factory = create_provider_factory()

        provider1 = factory("MOCK")
        provider2 = factory("Mock")
        provider3 = factory("mock")

        assert isinstance(provider1, MockLLMProvider)
        assert isinstance(provider2, MockLLMProvider)
        assert isinstance(provider3, MockLLMProvider)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_multiple_chained_operations(self):
        """Test multiple chained operations maintain state"""
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .cognitive_op(CognitiveOperation.ANALYZE)  # Override
                 .focus(FocusAspect.PATTERNS)
                 .focus(FocusAspect.STRUCTURE))  # Override

        assert prompt._cognitive_op == CognitiveOperation.ANALYZE
        assert prompt._focus == FocusAspect.STRUCTURE

    def test_deep_copy_in_fork(self):
        """Test that fork creates true deep copies"""
        prompt = ComposingPrompt()
        prompt._llm_augmentations.append({'type': 'test'})
        prompt._examples.append("Example")
        prompt._context_additions.append("Context")

        forks = prompt.fork(n=2)

        # Modify fork
        forks[0]._llm_augmentations.append({'type': 'new'})
        forks[0]._examples.append("New Example")

        # Original should be unchanged
        assert len(prompt._llm_augmentations) == 1
        assert len(prompt._examples) == 1
        assert len(forks[0]._llm_augmentations) == 2
        assert len(forks[0]._examples) == 2

    def test_chain_with_empty_context(self):
        """Test chaining with empty contexts"""
        prompt1 = ComposingPrompt()  # No context
        prompt2 = ComposingPrompt()

        chain = prompt1.chain(prompt2)

        # Should handle empty context gracefully
        assert prompt2._problem_context is None or prompt2._problem_context == ""

    def test_compose_with_invalid_strategy(self):
        """Test compose_with handles invalid merge strategy"""
        prompt1 = ComposingPrompt()
        prompt2 = ComposingPrompt()

        # Should not raise error for unknown strategy
        # Implementation might ignore or use default
        result = prompt1.compose_with(prompt2, merge_strategy='invalid')
        assert result is prompt1

    def test_parallel_execution_with_failures(self):
        """Test parallel execution handles provider failures"""
        failing_provider = Mock(spec=MockLLMProvider)
        failing_provider.generate.side_effect = Exception("API Error")

        working_provider = Mock(spec=MockLLMProvider)
        working_provider.generate.return_value = "Success"

        calls = [
            {'provider': failing_provider, 'instruction': "Test 1"},
            {'provider': working_provider, 'instruction': "Test 2"}
        ]

        prompt = ComposingPrompt()

        # Should handle mixed success/failure
        with pytest.raises(Exception):
            prompt._execute_parallel_llm_calls(calls)

    def test_termination_with_empty_state(self):
        """Test termination detection with empty state"""
        prompt = ComposingPrompt()
        result = prompt.llm_termination("")

        assert result is False  # Empty should not be terminal

    def test_termination_with_none_provider(self):
        """Test termination when no provider is set"""
        prompt = ComposingPrompt(_llm_provider=None)
        result = prompt.llm_termination("Some text")

        # Should fall back to pattern matching or length
        assert isinstance(result, bool)