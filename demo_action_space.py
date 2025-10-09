#!/usr/bin/env python3
"""
Compositional Prompting as an Action Space

This demo shows how compositional prompting provides a structured action space
that can be used by external systems like MCTS, RL agents, or other controllers.

The library doesn't implement MCTS/RL itself - it provides the compositional
framework that these systems can use to construct and explore prompts systematically.
"""

from compositional_prompting import (
    ComposingPrompt, MockLLMProvider,
    CognitiveOperation, FocusAspect, ReasoningStyle,
    ConnectionType, OutputFormat,
    MetaStrategy, ConfidenceLevel, ReasoningDepth
)
import random
import json


def demo_action_space_for_external_systems():
    """Show how external systems can use compositional prompting as an action space"""

    print("=" * 70)
    print("COMPOSITIONAL PROMPTING AS AN ACTION SPACE")
    print("=" * 70)
    print("\nThis library provides a structured way to represent prompts as")
    print("compositional actions that external systems (MCTS, RL, etc.) can use.\n")

    # ========================================================================
    # Example 1: Random Agent using the action space
    # ========================================================================
    print("1. RANDOM AGENT EXAMPLE")
    print("-" * 40)
    print("An external agent randomly sampling from the action space:\n")

    class RandomAgent:
        """Simulates an external agent that uses compositional prompting"""

        def __init__(self):
            self.llm = MockLLMProvider()

        def sample_action(self):
            """Sample a random compositional action"""
            return ComposingPrompt.sample_action(provider=self.llm)

        def get_action_with_bias(self, weights):
            """Sample with specific biases (e.g., from learned policy)"""
            return ComposingPrompt.sample_action(weights=weights, provider=self.llm)

    agent = RandomAgent()

    # Sample 3 random actions
    print("Sampling 3 random actions:")
    for i in range(3):
        action = agent.sample_action()
        vector = action.get_action_vector()
        print(f"  Action {i+1}: ω={vector['omega']}, φ={vector['phi']}, σ={vector['sigma']}")

    # ========================================================================
    # Example 2: RL Agent with learned weights
    # ========================================================================
    print("\n2. RL AGENT WITH LEARNED WEIGHTS")
    print("-" * 40)
    print("An RL agent that has learned to prefer certain actions:\n")

    # Simulated learned weights from Q-learning or policy gradient
    learned_weights = {
        'cognitive_op': {
            CognitiveOperation.DECOMPOSE: 3.0,  # Agent learned decomposition is effective
            CognitiveOperation.ANALYZE: 2.0,
            CognitiveOperation.VERIFY: 1.5,
            CognitiveOperation.GENERATE: 0.5,   # Less effective for this task
        },
        'focus': {
            FocusAspect.STRUCTURE: 2.5,
            FocusAspect.CONSTRAINTS: 2.0,
            FocusAspect.PATTERNS: 1.0,
        }
    }

    biased_action = agent.get_action_with_bias(learned_weights)
    vector = biased_action.get_action_vector()
    print(f"Biased sample: ω={vector['omega']}, φ={vector['phi']}")
    print("(More likely to be DECOMPOSE/ANALYZE with STRUCTURE focus)")

    # ========================================================================
    # Example 3: Action vectors for state representation
    # ========================================================================
    print("\n3. ACTION VECTORS FOR STATE REPRESENTATION")
    print("-" * 40)
    print("Converting compositional actions to vectors for ML models:\n")

    # Create a specific compositional action
    action = (ComposingPrompt()
        .cognitive_op(CognitiveOperation.ANALYZE)
        .focus(FocusAspect.PATTERNS)
        .style(ReasoningStyle.SYSTEMATIC)
        .meta_strategy(MetaStrategy.FORWARD_CHAINING)
        .confidence(ConfidenceLevel.HIGH)
        .depth(ReasoningDepth.DEEP))

    # Get structured vector (for tree search, RL state, etc.)
    vector = action.get_action_vector()
    print("Structured vector representation:")
    print(json.dumps(vector, indent=2))

    # This vector can be used by:
    # - MCTS for node representation
    # - RL for state/action encoding
    # - Neural networks for embedding

    # ========================================================================
    # Example 4: Compositional algebra for action construction
    # ========================================================================
    print("\n4. COMPOSITIONAL ALGEBRA FOR COMPLEX ACTIONS")
    print("-" * 40)
    print("Building complex actions through composition:\n")

    # Base components (like basis vectors)
    analyze = ComposingPrompt().cognitive_op(CognitiveOperation.ANALYZE)
    systematic = ComposingPrompt().style(ReasoningStyle.SYSTEMATIC)
    deep = ComposingPrompt().depth(ReasoningDepth.DEEP)
    patterns = ComposingPrompt().focus(FocusAspect.PATTERNS)

    # Compose them
    complex_action = analyze.compose_with(systematic).compose_with(deep).compose_with(patterns)

    print("Base components: ANALYZE, SYSTEMATIC, DEEP, PATTERNS")
    print("Composed action vector:")
    composed_vector = complex_action.get_action_vector()
    print(f"  ω={composed_vector['omega']}, σ={composed_vector['sigma']}, "
          f"φ={composed_vector['phi']}, depth=deep")

    # ========================================================================
    # Example 5: Action execution and reward calculation
    # ========================================================================
    print("\n5. ACTION EXECUTION FOR EXTERNAL SYSTEMS")
    print("-" * 40)
    print("How external systems execute actions and get rewards:\n")

    class ExternalController:
        """Simulates an external system using compositional prompting"""

        def __init__(self):
            self.llm = MockLLMProvider()

        def execute_action(self, state: str, action: ComposingPrompt) -> tuple:
            """Execute a compositional action on a state"""
            # Set the problem context
            action.problem_context(state)

            # Build and execute the prompt
            prompt = action.build()

            # In real system, this would call the LLM
            response = self.llm.generate(prompt)

            # Return new state and reward (simplified)
            new_state = f"{state}\n{response}"
            reward = self.calculate_reward(new_state)

            return new_state, reward

        def calculate_reward(self, state: str) -> float:
            """Calculate reward for the state (task-specific)"""
            # Simplified: reward based on length and keywords
            reward = min(len(state) / 1000, 1.0) * 0.3
            if "solution" in state.lower():
                reward += 0.5
            if "therefore" in state.lower():
                reward += 0.2
            return reward

    controller = ExternalController()
    initial_state = "Solve: Find the maximum of f(x) = -x² + 4x + 5"

    # Sample an action
    action = ComposingPrompt.sample_action(provider=controller.llm)

    # Execute it
    new_state, reward = controller.execute_action(initial_state, action)

    print(f"Initial state: {initial_state}")
    print(f"Action taken: {action.get_action_vector()['omega']}")
    print(f"Reward received: {reward:.3f}")


def demo_action_space_statistics():
    """Show the size and structure of the compositional action space"""

    print("\n" + "=" * 70)
    print("COMPOSITIONAL ACTION SPACE STATISTICS")
    print("=" * 70)

    # Calculate action space size
    core_dims = {
        'ω (Cognitive Op)': len(CognitiveOperation),
        'φ (Focus)': len(FocusAspect),
        'σ (Style)': len(ReasoningStyle),
        'κ (Connection)': len(ConnectionType),
        'τ (Output)': len(OutputFormat),
    }

    extended_dims = {
        'Meta Strategy': len(MetaStrategy),
        'Confidence': len(ConfidenceLevel),
        'Depth': len(ReasoningDepth),
    }

    print("\nCore Dimensions:")
    total_core = 1
    for name, size in core_dims.items():
        print(f"  {name:20} = {size} options")
        total_core *= size

    print(f"\nTotal core combinations: {total_core:,}")

    print("\nExtended Dimensions:")
    total_extended = total_core
    for name, size in extended_dims.items():
        print(f"  {name:20} = {size} options")
        total_extended *= size

    print(f"\nTotal with extensions: {total_extended:,}")

    print("\nKey Insight:")
    print("  Instead of {total_extended:,} distinct prompts,")
    print(f"  we have ~{sum(core_dims.values()) + sum(extended_dims.values())} orthogonal parameters")
    print("  This factorization enables efficient exploration by RL/MCTS agents!")


def demo_integration_patterns():
    """Show how different systems can integrate with compositional prompting"""

    print("\n" + "=" * 70)
    print("INTEGRATION PATTERNS FOR EXTERNAL SYSTEMS")
    print("=" * 70)

    print("\n1. MCTS INTEGRATION PATTERN:")
    print("-" * 40)
    print("""
    # In your MCTS implementation:

    class MCTSNode:
        def __init__(self, state):
            self.state = state
            self.action = None  # Will be a ComposingPrompt

        def expand(self):
            # Sample action from compositional space
            action = ComposingPrompt.sample_action(weights=self.learned_weights)
            new_state = self.execute(action)
            child = MCTSNode(new_state)
            child.action = action
            return child
    """)

    print("\n2. RL AGENT INTEGRATION PATTERN:")
    print("-" * 40)
    print("""
    # In your RL agent:

    class RLAgent:
        def __init__(self):
            self.q_values = {}  # Maps action vectors to Q-values

        def select_action(self, state, epsilon=0.1):
            if random.random() < epsilon:
                # Explore: sample from compositional space
                return ComposingPrompt.sample_action()
            else:
                # Exploit: use learned Q-values to weight sampling
                weights = self.compute_weights_from_q_values()
                return ComposingPrompt.sample_action(weights=weights)
    """)

    print("\n3. EVOLUTIONARY ALGORITHM PATTERN:")
    print("-" * 40)
    print("""
    # In your genetic algorithm:

    class Individual:
        def __init__(self):
            # Genome is a compositional action
            self.genome = ComposingPrompt.sample_action()

        def mutate(self):
            # Mutate by changing one dimension
            dimension = random.choice(['cognitive_op', 'focus', 'style'])
            if dimension == 'cognitive_op':
                self.genome.cognitive_op(random.choice(list(CognitiveOperation)))
    """)

    print("\n4. HUMAN-IN-THE-LOOP PATTERN:")
    print("-" * 40)
    print("""
    # For interactive systems:

    def interactive_reasoning(problem):
        base = ComposingPrompt().problem_context(problem)

        print("Select cognitive operation:")
        for i, op in enumerate(CognitiveOperation):
            print(f"  {i}. {op.value}")
        choice = int(input("Choice: "))

        base.cognitive_op(list(CognitiveOperation)[choice])
        return base.build()
    """)


def main():
    """Run all demonstrations"""

    print("\n" + "=" * 70)
    print("🧩 COMPOSITIONAL PROMPTING: A STRUCTURED ACTION SPACE FOR AI SYSTEMS")
    print("=" * 70)

    print("""
This library provides a compositional framework for representing prompts
as structured actions. It does NOT implement MCTS, RL, or other algorithms.

Instead, it offers:
- A factorized action space (ω, φ, σ, κ, τ + extensions)
- Fluid API for action construction
- Action vectors for state representation
- Weighted sampling for biased exploration
- Compositional algebra for complex actions

External systems (MCTS, RL, evolutionary algorithms, etc.) can use this
framework to systematically explore the space of possible prompts.
""")

    demo_action_space_for_external_systems()
    demo_action_space_statistics()
    demo_integration_patterns()

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY")
    print("=" * 70)
    print("""
Compositional Prompting is a LIBRARY, not an algorithm.
It provides the action space that algorithms can explore.

Think of it like OpenAI Gym:
- Gym provides environments and action spaces
- RL algorithms (DQN, PPO, etc.) use these spaces
- Similarly, compositional prompting provides prompt spaces
- MCTS/RL/other algorithms can explore these spaces
""")


if __name__ == "__main__":
    main()