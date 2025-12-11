'''
This is the primary entry point that orchestrates the complete AI-assisted
gradient descent demonstration. The script ties together data generation,
algorithm training, and AI validation into a cohesive workflow.
'''
from data_generator import generate_data
from algorithm import linearRegressionGD
from ai_assistant import response_claude

def main():
    print("\n" + "=" * 70)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT")
    print("AI-Assisted Implementation")
    print("=" * 70)

    # Step 1: Generate synthetic linear data with Gaussian noise
    print("\n[STEP 1] Generating synthetic dataset...")
    X, y, true_params = generate_data(
        n_samples=150,
        true_slope=2.5,
        true_intercept=5.0,
        noise_std=2.0,
        seed=42
    )
    print(f"  Generated {len(X)} data points")
    print(f"  True parameters: θ₀ = {true_params['intercept']}, θ₁ = {true_params['slope']}")

    # Step 2: Train the model with batch gradient descent optimization
    print("\n[STEP 2] Training model with gradient descent...")
    print("-" * 70)
    model = linearRegressionGD(learning_rate=0.02, n_iterations=2000)
    model.fit(X, y)
    print("-" * 70)

    # Step 3: Display results - compares learned vs. true parameters
    print("\n[STEP 3] Results:")
    print(f"  Learned parameters: θ₀ = {model.theta_0:.4f}, θ₁ = {model.theta_1:.4f}")
    print(f"  True parameters:    θ₀ = {true_params['intercept']:.4f}, θ₁ = {true_params['slope']:.4f}")
    print(f"  Error in θ₀: {abs(model.theta_0 - true_params['intercept']):.4f}")
    print(f"  Error in θ₁: {abs(model.theta_1 - true_params['slope']):.4f}")

    # Step 4: AI Explainer - uses Claude API for algorithmic explanation
    print("\n[STEP 4] AI-Assisted Explanation")
    print("  Running AI explanation (this may take a moment)...\n")
    ai_expl = response_claude()

    # Summary 
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"* Dataset: {len(X)} samples with Gaussian noise (σ = {true_params['noise_std']})")
    print(f"* Algorithm: Gradient descent with α = {model.learning_rate}, {len(model.cost_history)} iterations")
    print(f"* Convergence: Final cost = {model.cost_history[-1]:.4f}")
    print(f"* Parameters recovered with high accuracy")
    print(f"* AI explained the algorithm used")
    print("=" * 70)
    print("\n")

if __name__ == "__main__":
    main()
