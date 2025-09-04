#!/usr/bin/env python3
"""
Basic demo of compositional-prompting package
"""

from compositional_prompting import (
    ComposingPrompt, CognitiveOperation, FocusAspect, ReasoningStyle,
    MockLLMProvider
)


def main():
    """Main demo function"""
    print("🧩 Compositional Prompting Framework Demo")
    print("=" * 50)
    
    # Demo 1: Basic compositional prompt
    print(f"\n📝 Basic Compositional Prompt")
    print("-" * 30)
    
    prompt = (ComposingPrompt()
              .set_llm_provider(MockLLMProvider())
              .problem_context("Prove that the sum of angles in a triangle is 180°")
              .cognitive_op(CognitiveOperation.ANALYZE)
              .focus(FocusAspect.STRUCTURE)
              .style(ReasoningStyle.FORMAL)
              .build())
    
    print(f"Generated prompt:")
    print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
    
    # Demo 2: Weighted sampling
    print(f"\n🎲 Weighted Action Sampling")
    print("-" * 30)
    
    weights = {
        'cognitive_op': {
            CognitiveOperation.DECOMPOSE: 4.0,
            CognitiveOperation.VERIFY: 2.0,
            CognitiveOperation.ANALYZE: 1.0,
        }
    }
    
    print("Sampling 8 actions with DECOMPOSE bias:")
    ops = []
    for _ in range(8):
        action = ComposingPrompt.sample_action(weights=weights)
        vector = action.get_action_vector()
        ops.append(vector.get('omega'))
    
    from collections import Counter
    counts = Counter(ops)
    for op, count in counts.most_common():
        print(f"  {op}: {count}")
    
    # Demo 3: Parallel orchestration
    print(f"\n⚡ Parallel Orchestration")
    print("-" * 30)
    
    complex_prompt = (ComposingPrompt()
                      .set_llm_provider(MockLLMProvider())
                      .problem_context("Design an AI system for medical diagnosis")
                      .cognitive_op(CognitiveOperation.DECOMPOSE)
                      .llm_add_examples(n=3, parallel=True)
                      .llm_augment("consider ethical implications")
                      .llm_coherence_check()
                      .build())
    
    vector = ComposingPrompt().llm_add_examples(n=3, parallel=True).get_action_vector()
    print(f"Parallel operations enabled: {vector['has_parallel_ops']}")
    
    # Demo 4: Smart termination
    print(f"\n🏁 Intelligent Termination Detection")
    print("-" * 30)
    
    test_states = [
        "Based on the geometric proof, we conclude that the sum of angles is 180°. QED.",
        "Let's start by drawing a triangle and labeling the vertices...",
        "Therefore, the final answer is 180 degrees for any triangle."
    ]
    
    detector = ComposingPrompt().set_llm_provider(MockLLMProvider())
    for i, state in enumerate(test_states):
        is_terminal = detector.llm_termination(state)
        print(f"  State {i+1}: {'✅ TERMINAL' if is_terminal else '❌ CONTINUE'}")
    
    print(f"\n✨ Demo completed!")
    print(f"This framework can be used by:")
    print(f"- MCTS reasoning systems")
    print(f"- Multi-agent AI systems") 
    print(f"- Interactive reasoning tools")
    print(f"- Educational AI applications")


if __name__ == "__main__":
    main()