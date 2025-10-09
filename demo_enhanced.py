#!/usr/bin/env python3
"""
Comprehensive demonstration of the Compositional Prompting Enhancement Layer.

This demo shows how the enhancement layer augments the core compositional library
with intelligent, time-bounded operations while maintaining clean separation of concerns.
"""

import time
from typing import List, Dict, Any, Optional
from compositional_prompting import (
    ComposingPrompt,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
    MockLLMProvider,
    OllamaProvider
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
    time_bounded
)


def create_rich_example_graph() -> ExampleGraph:
    """Create a rich example graph with multiple domains and connections"""

    graph = ExampleGraph(similarity_threshold=0.6)

    # Mathematics examples
    math_examples = [
        Example(
            id="math_1",
            content="For absolute value inequalities |x| ≤ a, the solution is -a ≤ x ≤ a",
            domain="mathematics",
            metadata={"topic": "inequalities", "difficulty": "basic"}
        ),
        Example(
            id="math_2",
            content="To count lattice points, use systematic enumeration or generating functions",
            domain="mathematics",
            metadata={"topic": "combinatorics", "difficulty": "intermediate"}
        ),
        Example(
            id="math_3",
            content="Proof by induction: base case, inductive hypothesis, inductive step",
            domain="mathematics",
            metadata={"topic": "proof techniques", "difficulty": "basic"}
        ),
        Example(
            id="math_4",
            content="For optimization problems, check KKT conditions for constrained optima",
            domain="mathematics",
            metadata={"topic": "optimization", "difficulty": "advanced"}
        ),
    ]

    # Computer Science examples
    cs_examples = [
        Example(
            id="cs_1",
            content="Binary search requires sorted array, O(log n) time complexity",
            domain="computer_science",
            metadata={"topic": "algorithms", "difficulty": "basic"}
        ),
        Example(
            id="cs_2",
            content="Dynamic programming: identify overlapping subproblems and optimal substructure",
            domain="computer_science",
            metadata={"topic": "algorithms", "difficulty": "intermediate"}
        ),
        Example(
            id="cs_3",
            content="For distributed systems, consider CAP theorem: consistency, availability, partition tolerance",
            domain="computer_science",
            metadata={"topic": "systems", "difficulty": "advanced"}
        ),
        Example(
            id="cs_4",
            content="Graph algorithms: DFS for topological sort, BFS for shortest path in unweighted graphs",
            domain="computer_science",
            metadata={"topic": "algorithms", "difficulty": "intermediate"}
        ),
    ]

    # Physics examples
    physics_examples = [
        Example(
            id="phys_1",
            content="Conservation laws: energy, momentum, and angular momentum are conserved in closed systems",
            domain="physics",
            metadata={"topic": "mechanics", "difficulty": "basic"}
        ),
        Example(
            id="phys_2",
            content="For harmonic oscillators, solution is x(t) = A cos(ωt + φ) where ω = √(k/m)",
            domain="physics",
            metadata={"topic": "oscillations", "difficulty": "intermediate"}
        ),
    ]

    # Add all examples to graph
    for ex in math_examples + cs_examples + physics_examples:
        graph.add_example(ex)

    return graph


def create_comprehensive_reference_provider() -> CompositeReferenceProvider:
    """Create a reference provider with multiple knowledge sources"""

    provider = CompositeReferenceProvider()

    # Mathematical references
    math_refs = DictionaryReferenceProvider({
        "absolute value": {
            "definition": "The non-negative magnitude of a real number",
            "notation": "|x| = x if x ≥ 0, -x if x < 0",
            "properties": ["|xy| = |x||y|", "||x| - |y|| ≤ |x - y|"]
        },
        "lattice points": {
            "definition": "Points with integer coordinates",
            "notation": "(x, y) where x, y ∈ ℤ",
            "counting": "Use Pick's theorem for polygons"
        },
        "induction": {
            "definition": "Mathematical proof technique",
            "steps": ["Base case", "Inductive hypothesis", "Inductive step"],
            "variants": ["Strong induction", "Structural induction"]
        },
        "KKT conditions": {
            "definition": "Karush-Kuhn-Tucker optimality conditions",
            "usage": "Necessary conditions for constrained optimization",
            "components": ["Stationarity", "Primal feasibility", "Dual feasibility", "Complementary slackness"]
        }
    })

    # Computer Science references
    cs_refs = DictionaryReferenceProvider({
        "binary search": {
            "definition": "Divide-and-conquer search algorithm",
            "complexity": "O(log n) time, O(1) space",
            "requirements": "Sorted array or searchable structure"
        },
        "dynamic programming": {
            "definition": "Optimization technique using memoization",
            "requirements": ["Optimal substructure", "Overlapping subproblems"],
            "patterns": ["Top-down with memoization", "Bottom-up tabulation"]
        },
        "CAP theorem": {
            "definition": "Fundamental theorem in distributed systems",
            "statement": "Cannot guarantee all three: Consistency, Availability, Partition tolerance",
            "implications": "Must choose 2 out of 3 properties"
        },
        "topological sort": {
            "definition": "Linear ordering of vertices in DAG",
            "algorithms": ["DFS-based", "Kahn's algorithm"],
            "complexity": "O(V + E)"
        }
    })

    # Physics references
    physics_refs = DictionaryReferenceProvider({
        "conservation laws": {
            "definition": "Quantities that remain constant in isolated systems",
            "examples": ["Energy", "Momentum", "Angular momentum", "Charge"],
            "noether": "Each conservation law corresponds to a symmetry"
        },
        "harmonic oscillator": {
            "definition": "System with restoring force proportional to displacement",
            "equation": "F = -kx",
            "frequency": "ω = √(k/m)"
        }
    })

    # Add all providers with priorities
    provider.add_provider("mathematics", math_refs, priority=20)
    provider.add_provider("computer_science", cs_refs, priority=20)
    provider.add_provider("physics", physics_refs, priority=20)

    return provider


def create_advanced_template_library() -> TemplateLibrary:
    """Create a template library with sophisticated patterns"""

    library = TemplateLibrary()

    # Advanced mathematical templates
    library.add_template(DecompositionTemplate(
        name="inequality_solving",
        pattern=r"(?:solve|find).*(?:inequality|inequalities).*\|(.+?)\|",
        steps=[
            "Identify the type of absolute value inequality",
            "Consider cases based on sign of expression: {0}",
            "Solve for each case separately",
            "Find intersection of solution sets",
            "Verify boundary conditions"
        ],
        applicable_domains=["mathematics"],
        confidence=0.9
    ))

    library.add_template(DecompositionTemplate(
        name="counting_problems",
        pattern=r"(?:count|find|how many).*(?:number of|total)",
        steps=[
            "Identify what needs to be counted",
            "Determine if order matters (permutation vs combination)",
            "Check for repetitions and constraints",
            "Apply appropriate counting principle",
            "Consider using generating functions if complex"
        ],
        applicable_domains=["mathematics", "combinatorics"],
        confidence=0.85
    ))

    # Algorithm design templates
    library.add_template(DecompositionTemplate(
        name="recursive_algorithm",
        pattern=r"(?:design|implement).*(?:recursive|recursion)",
        steps=[
            "Define base case(s)",
            "Identify recursive case(s)",
            "Ensure progress toward base case",
            "Analyze time and space complexity",
            "Consider iterative alternative if stack overflow risk"
        ],
        applicable_domains=["computer_science", "algorithms"],
        confidence=0.9
    ))

    library.add_template(DecompositionTemplate(
        name="graph_algorithm",
        pattern=r"(?:find|compute|algorithm).*(?:graph|tree|network)",
        steps=[
            "Identify graph properties (directed, weighted, cyclic)",
            "Choose traversal method (DFS vs BFS)",
            "Design data structures for tracking state",
            "Implement main algorithm logic",
            "Handle edge cases (disconnected components, cycles)"
        ],
        applicable_domains=["computer_science", "algorithms"],
        confidence=0.88
    ))

    # Physics problem templates
    library.add_template(DecompositionTemplate(
        name="mechanics_problem",
        pattern=r"(?:find|calculate|determine).*(?:force|acceleration|velocity|position)",
        steps=[
            "Draw free body diagram",
            "Identify all forces and constraints",
            "Choose coordinate system",
            "Apply Newton's laws or conservation principles",
            "Solve differential equations if needed",
            "Check units and physical reasonableness"
        ],
        applicable_domains=["physics", "mechanics"],
        confidence=0.87
    ))

    return library


def demonstrate_time_bounded_operations():
    """Demonstrate graceful degradation with time bounds"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Time-Bounded Operations")
    print("="*60)

    print("\n1. Operations within time budget:")
    with time_bounded(100) as timer:
        print(f"   Starting with {timer.remaining_ms:.1f}ms")

        # Simulate quick operation
        time.sleep(0.02)  # 20ms
        print(f"   After quick op: {timer.remaining_ms:.1f}ms remaining")

        # Simulate medium operation
        time.sleep(0.03)  # 30ms
        print(f"   After medium op: {timer.remaining_ms:.1f}ms remaining")

        # Check if we have time for expensive operation
        if timer.remaining_ms > 40:
            print("   Have time for expensive operation")
            time.sleep(0.03)
        else:
            print("   Skipping expensive operation (insufficient time)")

        print(f"   Total elapsed: {timer.elapsed_ms:.1f}ms")

    print("\n2. Graceful degradation when time expires:")
    with time_bounded(50) as timer:
        operations_completed = []

        # Try multiple operations
        for i, delay in enumerate([0.01, 0.015, 0.02, 0.025]):
            if not timer.is_expired:
                time.sleep(delay)
                operations_completed.append(f"Op{i+1}")
                print(f"   Completed: {operations_completed[-1]} at {timer.elapsed_ms:.1f}ms")
            else:
                print(f"   Skipped Op{i+1} - time expired")

        print(f"   Completed {len(operations_completed)} of 4 operations")


def demonstrate_graph_based_rag():
    """Demonstrate sophisticated RAG with graph operations"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Graph-Based RAG System")
    print("="*60)

    graph = create_rich_example_graph()

    print(f"\n1. Graph Statistics:")
    print(f"   Total examples: {len(graph.examples)}")
    print(f"   Total edges: {len(graph.edges)}")

    # Community detection
    communities = graph.detect_communities()
    unique_communities = set(communities.values())
    print(f"   Communities detected: {len(unique_communities)}")

    for comm_id in unique_communities:
        members = [ex_id for ex_id, c in communities.items() if c == comm_id]
        domains = set(graph.examples[ex_id].domain for ex_id in members)
        print(f"     Community {comm_id}: {len(members)} examples from {domains}")

    # Hub detection
    print("\n2. Hub Examples (most connected):")
    hubs = graph.find_hubs(top_k=3)
    for hub in hubs:
        print(f"   - [{hub.domain}] {hub.content[:50]}...")

    # Bridge detection
    print("\n3. Bridge Examples (connecting communities):")
    bridges = graph.find_bridges()
    for bridge in bridges[:2]:
        print(f"   - [{bridge.domain}] {bridge.content[:50]}...")

    # Retrieval strategies
    query = "solve optimization problem with constraints"

    print(f"\n4. Retrieval Strategies for query: '{query}'")

    for strategy in ["similar", "diverse", "hub_aware", "community"]:
        examples = graph.retrieve_similar(query, k=2, strategy=strategy)
        print(f"\n   Strategy: {strategy}")
        for ex in examples:
            print(f"     - [{ex.domain}] {ex.content[:60]}...")


def demonstrate_fast_augmentation():
    """Demonstrate fast, local augmentation"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Fast Local Augmentation")
    print("="*60)

    augmenter = FastAugmenter()

    # Test various problems
    test_cases = [
        {
            "prompt": "Solve this step by step",
            "problem": "Find all integer solutions to |x| + |y| ≤ 5"
        },
        {
            "prompt": "Analyze the algorithm",
            "problem": "Implement QuickSort with pivot selection optimization"
        },
        {
            "prompt": "Derive the formula",
            "problem": "Calculate the moment of inertia for a rotating disk"
        }
    ]

    print("\n1. Context-Specific Augmentation:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n   Test {i}:")
        print(f"   Original: {case['prompt']}")
        augmented = augmenter.make_specific(case['prompt'], case['problem'])
        print(f"   Augmented: {augmented}")

    print("\n2. Decomposition Suggestions:")
    for problem in [
        "Prove that √2 is irrational",
        "Design a distributed cache with consistency guarantees"
    ]:
        print(f"\n   Problem: {problem}")
        steps = augmenter.suggest_decomposition(problem)
        print("   Suggested steps:")
        for step in steps:
            print(f"     - {step}")

    print("\n3. Constraint Identification:")
    problem = "Find the maximum value of f(x,y) = x²+y² subject to x+y ≤ 10 and x,y ≥ 0"
    constraints = augmenter.identify_constraints(problem)
    print(f"   Problem: {problem[:50]}...")
    print("   Identified constraints:")
    for category, items in constraints.items():
        if items:
            print(f"     {category}: {items}")


def demonstrate_reference_lookup():
    """Demonstrate pluggable reference system"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Pluggable Reference Lookup")
    print("="*60)

    provider = create_comprehensive_reference_provider()

    # Single term lookup
    print("\n1. Single Term Lookups:")
    terms = ["absolute value", "binary search", "conservation laws"]

    for term in terms:
        result = provider.lookup(term)
        if result:
            print(f"\n   {term}:")
            print(f"     Source: {result['source']}")
            print(f"     Definition: {result.get('definition', 'N/A')[:80]}...")

    # Batch lookup
    print("\n2. Batch Lookup with Context:")
    math_terms = ["induction", "KKT conditions", "lattice points"]
    results = provider.batch_lookup(math_terms, context="mathematical proof")

    for term, info in results.items():
        print(f"\n   {term} (from {info['source']}):")
        for key, value in info.items():
            if key != 'source' and value:
                if isinstance(value, list):
                    print(f"     {key}: {', '.join(str(v) for v in value[:2])}...")
                else:
                    print(f"     {key}: {str(value)[:60]}...")


def demonstrate_template_system():
    """Demonstrate decomposition templates"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Decomposition Templates")
    print("="*60)

    library = create_advanced_template_library()

    test_problems = [
        ("Solve the inequality |2x - 3| < 7", "mathematics"),
        ("Count the number of ways to arrange 5 books on 3 shelves", "combinatorics"),
        ("Design a recursive algorithm to find all permutations", "algorithms"),
        ("Find the shortest path in a weighted graph", "algorithms"),
        ("Calculate the force on a charged particle in a magnetic field", "physics")
    ]

    for problem, domain in test_problems:
        print(f"\n   Problem: {problem}")
        print(f"   Domain: {domain}")

        # Find matching templates
        templates = library.find_matching_templates(problem, domain)
        if templates:
            print(f"   Matched template: {templates[0].name}")
            steps = templates[0].apply(problem)
            print("   Decomposition:")
            for i, step in enumerate(steps, 1):
                print(f"     {i}. {step}")
        else:
            print("   No matching template found")


def demonstrate_full_integration():
    """Demonstrate the complete enhanced system"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Full System Integration")
    print("="*60)

    # Create fully configured orchestrator
    orchestrator = EnhancementOrchestrator(
        EnhancementConfig(
            enable_rag=True,
            enable_fast_augmentation=True,
            enable_reference_lookup=True,
            enable_templates=True,
            max_latency_ms=500
        )
    )

    # Configure components
    orchestrator.with_example_graph(create_rich_example_graph())
    orchestrator.with_reference_provider(create_comprehensive_reference_provider())
    orchestrator.with_template_library(create_advanced_template_library())

    # Test problem
    problem = "Prove that the sum of the first n positive integers equals n(n+1)/2"

    print(f"\n1. Problem: {problem}")

    # Create base compositional prompt
    print("\n2. Base Compositional Prompt:")
    base_prompt = (
        ComposingPrompt()
        .problem_context(problem)
        .cognitive_op(CognitiveOperation.VERIFY)
        .focus(FocusAspect.CORRECTNESS)
        .style(ReasoningStyle.FORMAL)
        .connect(ConnectionType.THEREFORE)
        .format(OutputFormat.STEPS)
    )
    print(f"   {base_prompt.build()[:200]}...")

    # Apply enhancements
    print("\n3. Applying Enhancements:")
    context = {"problem": problem}
    enhanced = orchestrator.enhance(base_prompt.build(), context)

    print("   Enhancements applied:")
    for enhancement_type, content in enhanced.get("enhancements", {}).items():
        print(f"     ✓ {enhancement_type}")
        if isinstance(content, list) and content:
            print(f"       - {content[0][:60]}...")
        elif isinstance(content, dict):
            print(f"       - {len(content)} items")
        elif isinstance(content, str):
            print(f"       - {content[:60]}...")

    # Use enhanced prompt API
    print("\n4. Enhanced Fluid API:")
    enhanced_prompt = (
        EnhancedComposingPrompt(orchestrator=orchestrator)
        .problem_context(problem)
        .cognitive_op(CognitiveOperation.VERIFY)
        .focus(FocusAspect.STRUCTURE)
        .style(ReasoningStyle.SYSTEMATIC)
        .with_smart_examples(n=2, strategy="diverse")
        .with_decomposition(domain="mathematics")
        .with_references(["induction"])
        .build_enhanced()
    )

    print(f"   Final enhanced prompt preview:")
    lines = enhanced_prompt.split('\n')
    for line in lines[:10]:
        if line:
            print(f"     {line[:70]}...")

    # Show metrics
    print("\n5. Performance Metrics:")
    metrics = orchestrator.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")


def demonstrate_parallel_exploration():
    """Demonstrate parallel prompt exploration with enhancements"""

    print("\n" + "="*60)
    print("DEMONSTRATION: Parallel Exploration")
    print("="*60)

    orchestrator = EnhancementOrchestrator()
    orchestrator.with_example_graph(create_rich_example_graph())

    problem = "Design an efficient sorting algorithm for nearly-sorted data"

    print(f"\n1. Problem: {problem}")

    # Create multiple exploration paths
    print("\n2. Creating parallel exploration paths:")

    strategies = [
        ("Analytical", CognitiveOperation.ANALYZE, FocusAspect.PATTERNS, ReasoningStyle.SYSTEMATIC),
        ("Creative", CognitiveOperation.GENERATE, FocusAspect.ALTERNATIVE_APPROACHES, ReasoningStyle.CREATIVE),
        ("Formal", CognitiveOperation.VERIFY, FocusAspect.CORRECTNESS, ReasoningStyle.FORMAL),
        ("Practical", CognitiveOperation.SYNTHESIZE, FocusAspect.EFFICIENCY, ReasoningStyle.INTUITIVE)
    ]

    for name, cog_op, focus, style in strategies:
        enhanced = (
            EnhancedComposingPrompt(orchestrator=orchestrator)
            .problem_context(problem)
            .cognitive_op(cog_op)
            .focus(focus)
            .style(style)
            .with_smart_examples(n=1, strategy="similar")
        )

        print(f"\n   {name} approach:")
        print(f"     Operation: {cog_op.value}")
        print(f"     Focus: {focus.value}")
        print(f"     Style: {style.value}")

        # Get action vector
        action_vector = enhanced.base.get_action_vector()
        print(f"     Action vector: ω={action_vector['omega']}, φ={action_vector['phi']}, σ={action_vector['sigma']}")


def main():
    """Run all demonstrations"""

    print("\n" + "="*70)
    print(" COMPOSITIONAL PROMPTING ENHANCEMENT LAYER - COMPREHENSIVE DEMO")
    print("="*70)
    print("\nThis demonstration showcases the elegant integration between the")
    print("pure compositional core and the intelligent enhancement layer.")
    print("\nKey principles demonstrated:")
    print("  • Time-bounded operations with graceful degradation")
    print("  • Graph-based RAG with community detection")
    print("  • Fast local augmentation without critical decisions")
    print("  • Pluggable reference systems")
    print("  • Deterministic decomposition templates")
    print("  • Clean separation of concerns")

    # Run all demonstrations
    demonstrate_time_bounded_operations()
    demonstrate_graph_based_rag()
    demonstrate_fast_augmentation()
    demonstrate_reference_lookup()
    demonstrate_template_system()
    demonstrate_full_integration()
    demonstrate_parallel_exploration()

    print("\n" + "="*70)
    print(" DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nThe enhancement layer provides:")
    print("  ✓ Optional intelligence without breaking determinism")
    print("  ✓ Time-bounded operations that fail gracefully")
    print("  ✓ Rich example retrieval through graph structures")
    print("  ✓ Pluggable backends for extensibility")
    print("  ✓ Beautiful, composable APIs following Unix philosophy")
    print("\nThe system maintains clean separation between the pure")
    print("compositional core and optional enhancements.")


if __name__ == "__main__":
    main()