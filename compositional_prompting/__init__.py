"""
Compositional Prompting Framework
A fluid API for building sophisticated prompts through compositional actions.
"""

from typing import List, Optional, Dict, Any, Union, Protocol
from enum import Enum
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import asyncio
import concurrent.futures
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Abstract LLM Provider Interface
class LLMProvider(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name"""
        pass


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        return f"[MOCK LLM Response to: {prompt[:50]}...]"
    
    def get_provider_name(self) -> str:
        return "MockLLM"


# Optional provider implementations
class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def get_provider_name(self) -> str:
        return f"OpenAI-{self.model}"


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def get_provider_name(self) -> str:
        return f"Anthropic-{self.model}"


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        try:
            import requests
            self.model = model
            self.base_url = base_url
            self.session = requests.Session()
        except ImportError:
            raise ImportError("requests package not installed. Install with: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        import requests
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            }
        )
        return response.json()["response"]
    
    def get_provider_name(self) -> str:
        return f"Ollama-{self.model}"


# Core Compositional Action Components
class CognitiveOperation(Enum):
    """ω: High-level reasoning approaches"""
    DECOMPOSE = "decompose"
    ANALYZE = "analyze"
    GENERATE = "generate"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"
    ABSTRACT = "abstract"


class FocusAspect(Enum):
    """φ: What to concentrate on"""
    STRUCTURE = "structure"
    CONSTRAINTS = "constraints"
    PATTERNS = "patterns"
    SOLUTION = "solution"
    CORRECTNESS = "correctness"
    EFFICIENCY = "efficiency"
    EXAMPLES = "examples"
    RELATIONSHIPS = "relationships"


class ReasoningStyle(Enum):
    """σ: How to approach the task"""
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"
    CRITICAL = "critical"
    FORMAL = "formal"
    INTUITIVE = "intuitive"


class ConnectionType(Enum):
    """κ: How to link to previous reasoning"""
    THEREFORE = "therefore"
    HOWEVER = "however"
    BUILDING_ON = "building_on"
    ALTERNATIVELY = "alternatively"
    VERIFY = "verify"
    CONTINUE = "continue"


class OutputFormat(Enum):
    """τ: How to structure the response"""
    STEPS = "steps"
    LIST = "list"
    MATHEMATICAL = "mathematical"
    NARRATIVE = "narrative"
    CODE = "code"
    SOLUTION = "solution"
    EXPLANATION = "explanation"
    TABLE = "table"


@dataclass
class ComposingPrompt:
    """
    Fluid API for building compositional prompts.
    
    Usage:
        llm = MockLLMProvider()  # or OpenAIProvider(api_key), etc.
        prompt = (ComposingPrompt(llm_provider=llm)
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .focus(FocusAspect.STRUCTURE)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .connect(ConnectionType.THEREFORE)
                 .format(OutputFormat.STEPS)
                 .llm_augment("make this specific to geometry")
                 .build())
    """
    
    # LLM provider for meta-reasoning
    _llm_provider: Optional[LLMProvider] = None
    
    # Core compositional components
    _cognitive_op: Optional[CognitiveOperation] = None
    _focus: Optional[FocusAspect] = None
    _style: Optional[ReasoningStyle] = None
    _connection: Optional[ConnectionType] = None
    _output_format: Optional[OutputFormat] = None
    
    # LLM enhancement chain
    _llm_augmentations: List[Dict[str, Any]] = field(default_factory=list)
    _examples: List[str] = field(default_factory=list)
    _coherence_checks: bool = False
    _coherence_provider: Optional[LLMProvider] = None
    _context_additions: List[str] = field(default_factory=list)
    
    # Base prompt content
    _base_prompt: str = ""
    _problem_context: str = ""
    
    def __post_init__(self):
        # Default to mock provider if none provided
        if self._llm_provider is None:
            self._llm_provider = MockLLMProvider()
    
    # Provider configuration methods
    def set_llm_provider(self, provider: LLMProvider) -> 'ComposingPrompt':
        """Set the LLM provider for meta-reasoning operations"""
        self._llm_provider = provider
        return self
    
    # Core compositional action methods
    def cognitive_op(self, operation: Union[CognitiveOperation, str]) -> 'ComposingPrompt':
        """Set the cognitive operation (ω)"""
        if isinstance(operation, str):
            operation = CognitiveOperation(operation)
        self._cognitive_op = operation
        return self
    
    def focus(self, aspect: Union[FocusAspect, str]) -> 'ComposingPrompt':
        """Set the focus aspect (φ)"""
        if isinstance(aspect, str):
            aspect = FocusAspect(aspect)
        self._focus = aspect
        return self
    
    def style(self, reasoning_style: Union[ReasoningStyle, str]) -> 'ComposingPrompt':
        """Set the reasoning style (σ)"""
        if isinstance(reasoning_style, str):
            reasoning_style = ReasoningStyle(reasoning_style)
        self._style = reasoning_style
        return self
    
    def connect(self, connection_type: Union[ConnectionType, str]) -> 'ComposingPrompt':
        """Set the connection type (κ)"""
        if isinstance(connection_type, str):
            connection_type = ConnectionType(connection_type)
        self._connection = connection_type
        return self
    
    def format(self, output_format: Union[OutputFormat, str]) -> 'ComposingPrompt':
        """Set the output format (τ)"""
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)
        self._output_format = output_format
        return self
    
    def problem_context(self, context: str) -> 'ComposingPrompt':
        """Set the problem context"""
        self._problem_context = context
        return self
    
    def base_prompt(self, prompt: str) -> 'ComposingPrompt':
        """Set the base prompt content"""
        self._base_prompt = prompt
        return self
    
    # LLM meta-reasoning enhancement methods
    def llm_augment(self, instruction: str, provider: Optional[LLMProvider] = None) -> 'ComposingPrompt':
        """Add LLM-based augmentation instruction"""
        self._llm_augmentations.append({
            'type': 'augment',
            'instruction': instruction,
            'provider': provider or self._llm_provider
        })
        return self
    
    def llm_coherence_check(self, provider: Optional[LLMProvider] = None) -> 'ComposingPrompt':
        """Enable LLM-based coherence checking"""
        self._coherence_checks = True
        self._coherence_provider = provider or self._llm_provider
        return self
    
    def llm_add_examples(self, n: int = 2, domain: str = None, provider: Optional[LLMProvider] = None, 
                        parallel: bool = True) -> 'ComposingPrompt':
        """Request LLM to generate relevant examples"""
        if domain:
            instruction = f"Generate {n} relevant examples from the {domain} domain"
        else:
            instruction = f"Generate {n} relevant examples"
            
        self._llm_augmentations.append({
            'type': 'examples',
            'instruction': instruction,
            'n': n,
            'domain': domain,
            'provider': provider or self._llm_provider,
            'parallel': parallel
        })
        return self
    
    def llm_termination(self, state: str, provider: Optional[LLMProvider] = None) -> bool:
        """Use LLM to determine if state represents a terminal/complete response"""
        import re
        
        # First check classical termination patterns
        termination_patterns = [
            r'\bfinal answer:?\s*(.+)', r'\bconclusion:?\s*(.+)',
            r'\btherefore,?\s+the answer is\s+(.+)', r'\bso,?\s+the solution is\s+(.+)',
            r'\bhence,?\s+(.+)', r'\bin conclusion,?\s+(.+)',
            r'\bthe result is\s+(.+)', r'\bthe answer is\s+(.+)',
            r'\bwe conclude that\s+(.+)', r'\bthus,?\s+(.+)',
            r'\bQED\b', r'\bproven\b', r'\bsolved\b',
            r'##\s*(answer|solution|conclusion)'
        ]
        
        for pattern in termination_patterns:
            if re.search(pattern, state.lower(), re.IGNORECASE):
                return True
        
        # LLM-based check for subtle completeness
        termination_prompt = f"""
Analyze this reasoning output and determine if it represents a complete answer.

Consider:
- Does it provide a definitive answer?
- Is the reasoning chain complete?
- Would a human consider this final?

Respond with only "TERMINAL" or "CONTINUE".

Output to analyze:
{state[-500:]}

Response:"""
        
        try:
            llm = provider or self._llm_provider
            if llm:
                response = llm.generate(termination_prompt, max_tokens=10, temperature=0.1)
                return "TERMINAL" in response.upper()
        except Exception:
            pass
        
        return len(state) > 2000  # Fallback to length check
    
    def rag_add_examples(self, n: int = 5, similarity_threshold: float = 0.8) -> 'ComposingPrompt':
        """Add examples using RAG based on context similarity"""
        # Placeholder for RAG implementation
        rag_instruction = f"Retrieve {n} similar examples (similarity > {similarity_threshold})"
        self._context_additions.append(rag_instruction)
        return self
    
    def add_context(self, context: str) -> 'ComposingPrompt':
        """Add additional context"""
        self._context_additions.append(context)
        return self
    
    def sample_weighted(self, weights: Optional[Dict[str, Dict[Any, float]]] = None) -> 'ComposingPrompt':
        """Sample action components using optional weights for biased exploration"""
        import random
        
        def _sample_weighted_component(options: List[Any], component_weights: Dict[Any, float]) -> Any:
            if not component_weights:
                return random.choice(options)
            
            # Normalize weights
            total_weight = sum(component_weights.get(opt, 1.0) for opt in options)
            probs = [component_weights.get(opt, 1.0) / total_weight for opt in options]
            
            return random.choices(options, weights=probs)[0]
        
        if weights:
            # Sample each component with weights
            if 'cognitive_op' in weights:
                self._cognitive_op = _sample_weighted_component(
                    list(CognitiveOperation), weights['cognitive_op']
                )
            
            if 'focus' in weights:
                self._focus = _sample_weighted_component(
                    list(FocusAspect), weights['focus']
                )
            
            if 'style' in weights:
                self._style = _sample_weighted_component(
                    list(ReasoningStyle), weights['style']
                )
            
            if 'connection' in weights:
                self._connection = _sample_weighted_component(
                    list(ConnectionType), weights['connection']
                )
            
            if 'output_format' in weights:
                self._output_format = _sample_weighted_component(
                    list(OutputFormat), weights['output_format']
                )
        else:
            # Uniform random sampling
            self._cognitive_op = random.choice(list(CognitiveOperation))
            self._focus = random.choice(list(FocusAspect))
            self._style = random.choice(list(ReasoningStyle))
            self._connection = random.choice(list(ConnectionType))
            self._output_format = random.choice(list(OutputFormat))
        
        return self
    
    @classmethod
    def sample_action(cls, weights: Optional[Dict[str, Dict[Any, float]]] = None,
                     provider: Optional[LLMProvider] = None) -> 'ComposingPrompt':
        """Factory method to create a randomly sampled compositional action"""
        return cls().set_llm_provider(provider).sample_weighted(weights)
    
    # Build methods
    def _build_compositional_core(self) -> str:
        """Build the core compositional prompt from (ω,φ,σ,κ,τ)"""
        parts = []
        
        if self._cognitive_op:
            if self._cognitive_op == CognitiveOperation.DECOMPOSE:
                parts.append("Let me break this problem down systematically.")
            elif self._cognitive_op == CognitiveOperation.ANALYZE:
                parts.append("Let me analyze this problem carefully.")
            elif self._cognitive_op == CognitiveOperation.GENERATE:
                parts.append("Let me generate a solution approach.")
            elif self._cognitive_op == CognitiveOperation.VERIFY:
                parts.append("Let me verify this reasoning step by step.")
            elif self._cognitive_op == CognitiveOperation.SYNTHESIZE:
                parts.append("Let me synthesize the key insights.")
            elif self._cognitive_op == CognitiveOperation.ABSTRACT:
                parts.append("Let me abstract the essential patterns.")
        
        if self._focus:
            if self._focus == FocusAspect.STRUCTURE:
                parts.append("I'll focus on the structural relationships and organization.")
            elif self._focus == FocusAspect.CONSTRAINTS:
                parts.append("I'll focus on the constraints and limitations.")
            elif self._focus == FocusAspect.PATTERNS:
                parts.append("I'll focus on identifying key patterns.")
            elif self._focus == FocusAspect.SOLUTION:
                parts.append("I'll focus on developing a clear solution.")
            elif self._focus == FocusAspect.CORRECTNESS:
                parts.append("I'll focus on ensuring correctness.")
            elif self._focus == FocusAspect.EFFICIENCY:
                parts.append("I'll focus on efficiency and optimization.")
        
        if self._style:
            if self._style == ReasoningStyle.SYSTEMATIC:
                parts.append("I'll approach this systematically and methodically.")
            elif self._style == ReasoningStyle.CREATIVE:
                parts.append("I'll approach this with creative thinking.")
            elif self._style == ReasoningStyle.CRITICAL:
                parts.append("I'll approach this with critical analysis.")
            elif self._style == ReasoningStyle.FORMAL:
                parts.append("I'll approach this with formal rigor.")
            elif self._style == ReasoningStyle.INTUITIVE:
                parts.append("I'll approach this with intuitive reasoning.")
        
        if self._connection:
            if self._connection == ConnectionType.THEREFORE:
                parts.append("Therefore,")
            elif self._connection == ConnectionType.HOWEVER:
                parts.append("However,")
            elif self._connection == ConnectionType.BUILDING_ON:
                parts.append("Building on the previous analysis,")
            elif self._connection == ConnectionType.ALTERNATIVELY:
                parts.append("Alternatively,")
            elif self._connection == ConnectionType.VERIFY:
                parts.append("To verify this,")
            elif self._connection == ConnectionType.CONTINUE:
                parts.append("Continuing from where we left off,")
        
        if self._output_format:
            if self._output_format == OutputFormat.STEPS:
                parts.append("I'll present this as clear steps:")
            elif self._output_format == OutputFormat.LIST:
                parts.append("I'll present this as a structured list:")
            elif self._output_format == OutputFormat.MATHEMATICAL:
                parts.append("I'll present this with mathematical notation:")
            elif self._output_format == OutputFormat.NARRATIVE:
                parts.append("I'll present this as a clear narrative:")
            elif self._output_format == OutputFormat.CODE:
                parts.append("I'll present this as code:")
            elif self._output_format == OutputFormat.SOLUTION:
                parts.append("Here's the solution:")
        
        return " ".join(parts)
    
    def build(self) -> str:
        """Build the final prompt"""
        components = []
        
        # Add base prompt if provided
        if self._base_prompt:
            components.append(self._base_prompt)
        
        # Add problem context if provided
        if self._problem_context:
            components.append(f"Problem: {self._problem_context}")
        
        # Add compositional core
        core = self._build_compositional_core()
        if core:
            components.append(core)
        
        # Add context additions (including RAG)
        if self._context_additions:
            components.extend(self._context_additions)
        
        # Add examples if any
        if self._examples:
            components.append("Relevant examples:")
            components.extend(self._examples)
        
        # Note: LLM augmentations would typically be applied
        # after the initial prompt generation in a real implementation
        if self._llm_augmentations:
            aug_descriptions = []
            for aug in self._llm_augmentations:
                if isinstance(aug, dict):
                    desc = aug['instruction']
                    if aug.get('provider'):
                        desc += f" (via {aug['provider'].get_provider_name()})"
                    aug_descriptions.append(desc)
                else:
                    aug_descriptions.append(str(aug))
            components.append(f"[LLM Augmentations to apply: {', '.join(aug_descriptions)}]")
        
        if self._coherence_checks:
            components.append("[Apply coherence checking]")
        
        return "\n\n".join(components)
    
    # Parallel execution methods
    def _execute_parallel_llm_calls(self, calls: List[Dict[str, Any]], max_workers: int = 4) -> List[str]:
        """Execute multiple LLM calls in parallel"""
        def make_call(call_info):
            provider = call_info['provider']
            instruction = call_info['instruction']
            return provider.generate(instruction)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_call, call) for call in calls]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results
    
    def execute_llm_pipeline(self, max_workers: int = 4) -> 'ComposingPrompt':
        """Execute all LLM operations in the pipeline"""
        if not self._llm_augmentations:
            return self
        
        # Group operations for parallel execution
        parallel_calls = []
        sequential_calls = []
        
        for aug in self._llm_augmentations:
            if aug.get('parallel', True) and aug['type'] == 'examples':
                # Break examples into individual calls for parallel execution
                n = aug.get('n', 1)
                for i in range(n):
                    parallel_calls.append({
                        'provider': aug['provider'],
                        'instruction': f"Generate 1 relevant example" + (f" from the {aug['domain']} domain" if aug['domain'] else ""),
                        'type': 'example'
                    })
            else:
                sequential_calls.append(aug)
        
        # Execute parallel calls
        if parallel_calls:
            logger.info(f"Executing {len(parallel_calls)} LLM calls in parallel")
            parallel_results = self._execute_parallel_llm_calls(parallel_calls, max_workers)
            self._examples.extend(parallel_results)
        
        # Execute sequential calls
        for call in sequential_calls:
            result = call['provider'].generate(call['instruction'])
            if call['type'] == 'augment':
                self._context_additions.append(f"Augmentation: {result}")
        
        # Execute coherence check if requested
        if self._coherence_checks and self._coherence_provider:
            current_prompt = self.build()
            coherence_result = self._coherence_provider.generate(
                f"Review this prompt for coherence and suggest improvements:\n\n{current_prompt}"
            )
            self._context_additions.append(f"Coherence Check: {coherence_result}")
        
        return self
    
    def get_action_vector(self) -> Dict[str, Any]:
        """Get the compositional action as a structured vector"""
        return {
            'omega': self._cognitive_op.value if self._cognitive_op else None,
            'phi': self._focus.value if self._focus else None,
            'sigma': self._style.value if self._style else None,
            'kappa': self._connection.value if self._connection else None,
            'tau': self._output_format.value if self._output_format else None,
            'llm_augmentations': len(self._llm_augmentations),
            'coherence_checks': self._coherence_checks,
            'context_additions': len(self._context_additions),
            'has_parallel_ops': any(aug.get('parallel', False) for aug in self._llm_augmentations)
        }


def demo_usage():
    """Demonstrate the fluid API usage"""
    
    # Initialize LLM provider (using mock for demo)
    llm = MockLLMProvider()
    
    # Example 1: Mathematical problem with systematic approach
    math_prompt = (
        ComposingPrompt()
        .set_llm_provider(llm)
        .problem_context("Find the number of integer solutions to |x| + |y| ≤ 5")
        .cognitive_op(CognitiveOperation.DECOMPOSE)
        .focus(FocusAspect.STRUCTURE)
        .style(ReasoningStyle.SYSTEMATIC)
        .connect(ConnectionType.THEREFORE)
        .format(OutputFormat.STEPS)
        .llm_add_examples(n=2, domain="combinatorics")
        .llm_coherence_check()
        .build()
    )
    
    print("=== Mathematical Problem Prompt ===")
    print(math_prompt)
    print("\nAction Vector:", ComposingPrompt().problem_context("Find the number of integer solutions to |x| + |y| ≤ 5").cognitive_op(CognitiveOperation.DECOMPOSE).focus(FocusAspect.STRUCTURE).get_action_vector())
    
    # Example 2: Creative problem with RAG examples  
    creative_prompt = (
        ComposingPrompt()
        .set_llm_provider(llm)
        .problem_context("Design a novel algorithm for real-time collaboration")
        .cognitive_op(CognitiveOperation.GENERATE)
        .focus(FocusAspect.PATTERNS)
        .style(ReasoningStyle.CREATIVE)
        .format(OutputFormat.NARRATIVE)
        .rag_add_examples(n=3, similarity_threshold=0.7)
        .llm_augment("consider distributed systems principles")
        .llm_augment("ensure low latency requirements")
        .build()
    )
    
    print("\n=== Creative Problem Prompt ===")
    print(creative_prompt)
    
    # Example 3: Show different providers
    print("\n=== Provider Comparison ===")
    providers = [
        MockLLMProvider(),
        # Uncomment these when you have API keys:
        # OpenAIProvider(api_key="your-key"),
        # AnthropicProvider(api_key="your-key"),
        # OllamaProvider(model="llama2")
    ]
    
    for provider in providers:
        print(f"\nUsing {provider.get_provider_name()}:")
        simple_prompt = (
            ComposingPrompt()
            .set_llm_provider(provider)
            .cognitive_op(CognitiveOperation.ANALYZE)
            .focus(FocusAspect.PATTERNS)
            .build()
        )
        print(f"Built prompt: {simple_prompt}")


def create_provider_factory():
    """Factory function to create providers based on config"""
    
    def create_provider(provider_type: str, **kwargs) -> LLMProvider:
        if provider_type.lower() == "mock":
            return MockLLMProvider()
        elif provider_type.lower() == "openai":
            return OpenAIProvider(**kwargs)
        elif provider_type.lower() == "anthropic":
            return AnthropicProvider(**kwargs)
        elif provider_type.lower() == "ollama":
            return OllamaProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    return create_provider


if __name__ == "__main__":
    demo_usage()