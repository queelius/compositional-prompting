#!/usr/bin/env python3
"""
Complete Walkthrough: From Problem to Prompt

This example shows step-by-step how compositional prompting works,
and explores whether it makes sense without LLM intermediate steps.
"""

from compositional_prompting import (
    ComposingPrompt, MockLLMProvider,
    CognitiveOperation, FocusAspect, ReasoningStyle,
    ConnectionType, OutputFormat,
    MetaStrategy, ConfidenceLevel, ReasoningDepth
)


def walkthrough_single_shot():
    """
    Scenario 1: Single-shot composition (no intermediate LLM calls)
    This is what the current library does.
    """
    print("=" * 70)
    print("SCENARIO 1: SINGLE-SHOT COMPOSITION")
    print("=" * 70)

    # Start with a problem
    problem = "Prove that for any triangle, the sum of its angles equals 180°"
    print(f"\nInitial Problem:\n{problem}\n")

    # Build a compositional prompt in one shot
    prompt = (ComposingPrompt()
        .problem_context(problem)
        .cognitive_op(CognitiveOperation.DECOMPOSE)
        .focus(FocusAspect.STRUCTURE)
        .style(ReasoningStyle.SYSTEMATIC)
        .connect(ConnectionType.THEREFORE)
        .format(OutputFormat.STEPS)
        .build())

    print("Generated Prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # Analyze what we got
    print("\n🤔 Analysis:")
    print("- The prompt is essentially a 'meta-instruction' template")
    print("- It tells the LLM HOW to approach the problem")
    print("- But it's generic - not specific to triangles or geometry")
    print("- The actual reasoning happens in the LLM, not in our composition")

    return prompt


def walkthrough_iterative():
    """
    Scenario 2: What if we had iterative composition with LLM in the loop?
    This explores your concern about needing LLM intermediate steps.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: ITERATIVE COMPOSITION (HYPOTHETICAL)")
    print("=" * 70)

    problem = "Prove that for any triangle, the sum of its angles equals 180°"
    print(f"\nInitial Problem:\n{problem}\n")

    # This is what an iterative system might do (MCTS/RL agent)
    print("Step-by-step with LLM in the loop:")
    print("-" * 40)

    # Step 1: Decompose
    print("\n1. ACTION: DECOMPOSE + STRUCTURE")
    prompt1 = (ComposingPrompt()
        .problem_context(problem)
        .cognitive_op(CognitiveOperation.DECOMPOSE)
        .focus(FocusAspect.STRUCTURE)
        .build())
    print(f"   Prompt: {prompt1[:100]}...")
    print("   LLM Output: [Would decompose into: parallel lines, transversals, angle relationships]")

    # Step 2: Analyze (based on decomposition)
    print("\n2. ACTION: ANALYZE + PATTERNS")
    # Now the context has changed! It includes the decomposition
    new_context = problem + "\nDecomposition: parallel lines, transversals..."
    prompt2 = (ComposingPrompt()
        .problem_context(new_context)
        .cognitive_op(CognitiveOperation.ANALYZE)
        .focus(FocusAspect.PATTERNS)
        .build())
    print(f"   Prompt: {prompt2[:100]}...")
    print("   LLM Output: [Would identify: alternate interior angles pattern]")

    # Step 3: Verify
    print("\n3. ACTION: VERIFY + CORRECTNESS")
    newer_context = new_context + "\nPattern: alternate interior angles..."
    prompt3 = (ComposingPrompt()
        .problem_context(newer_context)
        .cognitive_op(CognitiveOperation.VERIFY)
        .focus(FocusAspect.CORRECTNESS)
        .build())
    print(f"   Prompt: {prompt3[:100]}...")
    print("   LLM Output: [Would verify: yes, angles sum to 180°]")

    print("\n🤔 Analysis:")
    print("- Each action builds on previous LLM outputs")
    print("- The context evolves based on reasoning progress")
    print("- Actions make more sense when they respond to current state")
    print("- This is really a JOINT action space: (context, action) → new_context")


def walkthrough_template_vs_reasoning():
    """
    Scenario 3: Examining the template vs actual reasoning distinction
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: TEMPLATE GUIDANCE VS ACTUAL REASONING")
    print("=" * 70)

    problem = "Design an efficient algorithm for finding the median of a stream"

    # Try different compositional strategies
    strategies = [
        {
            'name': 'Systematic Decomposition',
            'ops': {
                'cognitive_op': CognitiveOperation.DECOMPOSE,
                'style': ReasoningStyle.SYSTEMATIC,
                'focus': FocusAspect.STRUCTURE
            }
        },
        {
            'name': 'Creative Generation',
            'ops': {
                'cognitive_op': CognitiveOperation.GENERATE,
                'style': ReasoningStyle.CREATIVE,
                'focus': FocusAspect.ALTERNATIVE_APPROACHES
            }
        },
        {
            'name': 'Pattern Analysis',
            'ops': {
                'cognitive_op': CognitiveOperation.ANALYZE,
                'style': ReasoningStyle.FORMAL,
                'focus': FocusAspect.PATTERNS
            }
        }
    ]

    print(f"\nProblem: {problem}\n")

    for strategy in strategies:
        print(f"\nStrategy: {strategy['name']}")
        print("-" * 40)

        prompt = ComposingPrompt().problem_context(problem)
        for key, value in strategy['ops'].items():
            getattr(prompt, key)(value)

        built = prompt.build()
        print(f"Template Generated:\n{built}\n")

        # What would actually happen
        print("What the LLM would do:")
        if strategy['name'] == 'Systematic Decomposition':
            print("  → Break into: online vs offline, exact vs approximate, space constraints")
        elif strategy['name'] == 'Creative Generation':
            print("  → Explore: two-heap approach, balanced BST, reservoir sampling variants")
        elif strategy['name'] == 'Pattern Analysis':
            print("  → Identify: order statistics pattern, streaming pattern, percentile pattern")

        print("\nInsight: The compositional action provides a 'thinking strategy'")
        print("         but the actual domain reasoning happens in the LLM")


def analyze_joint_action_space():
    """
    Analysis: Is this really a joint action space?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: THE JOINT ACTION SPACE NATURE")
    print("=" * 70)

    print("""
The compositional prompting library provides:

1. TEMPLATE ACTIONS (What we have now):
   - (ω, φ, σ, κ, τ) → template
   - Templates are domain-independent
   - They guide HOW to think, not WHAT to think about

2. JOINT ACTIONS (What actually happens in practice):
   - (context, ω, φ, σ, κ, τ) → prompt → LLM → new_context
   - The action's effect depends on current context
   - The LLM fills in domain-specific reasoning

3. KEY INSIGHTS:

   a) Compositional prompting is like providing "cognitive scaffolding":
      - DECOMPOSE doesn't decompose anything itself
      - It tells the LLM to decompose
      - The quality depends on the LLM's capabilities

   b) The framework makes most sense for ITERATIVE systems:
      - MCTS: Each node has different context, same action has different effects
      - RL: State includes context, action effect is context-dependent
      - Single-shot: Limited to generic template guidance

   c) The "action space" is really a "meta-action space":
      - Actions are strategies, not direct manipulations
      - Like choosing "use hammer" vs "use screwdriver"
      - The actual work happens when the tool meets the material

4. THIS IS ACTUALLY GOOD:

   - Separation of concerns is maintained
   - The library provides structure without domain knowledge
   - External systems (MCTS/RL) handle the iteration and context evolution
   - The LLM provides domain expertise

   Architecture:
   [Context] → [Compositional Action] → [Prompt Template] → [LLM] → [New Context]
      ↑                                                                      ↓
      └──────────────── External System (MCTS/RL) ────────────────────────┘
""")


def demonstrate_context_evolution():
    """
    Show how context evolution would work in practice
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: CONTEXT EVOLUTION IN PRACTICE")
    print("=" * 70)

    print("""
How an MCTS system would use compositional prompting:

```python
class MCTSNode:
    def __init__(self, context: str):
        self.context = context  # Current reasoning state
        self.children = []

    def expand(self):
        # Sample compositional action
        action = ComposingPrompt.sample_action(
            weights=self.get_context_dependent_weights()
        )

        # Apply action to current context
        prompt = action.problem_context(self.context).build()

        # Execute with LLM
        llm_response = llm.generate(prompt)

        # Create new context (accumulates reasoning)
        new_context = f"{self.context}\\n{llm_response}"

        # Create child node with evolved context
        child = MCTSNode(new_context)
        self.children.append(child)

        return child
```

The key is that:
1. Each node has a different context (accumulated reasoning)
2. Same compositional action produces different effects at different nodes
3. The tree explores different reasoning paths
4. Context evolution is handled by MCTS, not the library
""")


def main():
    print("\n" + "=" * 70)
    print("🔍 COMPLETE WALKTHROUGH: COMPOSITIONAL PROMPTING IN PRACTICE")
    print("=" * 70)

    # Run all scenarios
    walkthrough_single_shot()
    walkthrough_iterative()
    walkthrough_template_vs_reasoning()
    analyze_joint_action_space()
    demonstrate_context_evolution()

    print("\n" + "=" * 70)
    print("FINAL THOUGHTS")
    print("=" * 70)

    print("""
Your intuition is correct:

1. The compositional framework provides "meta-actions" or "thinking strategies"
   - Not direct reasoning, but templates for reasoning
   - Domain-independent cognitive scaffolding

2. It IS a joint action space in practice:
   - (context, compositional_action) → new_context
   - Same action, different context = different outcome

3. This is the RIGHT design:
   - Library stays pure (just templates)
   - External systems handle context and iteration
   - LLM provides domain expertise

4. For single-shot use, it's limited to generic guidance
   For iterative use (MCTS/RL), it enables systematic exploration

The library is like providing a "cognitive toolkit" where:
- The tools are domain-independent strategies
- The external system decides which tool to use when
- The LLM does the actual work with the tool
- The result depends on both the tool chosen and the material (context)
""")


if __name__ == "__main__":
    main()