'''
This module integrates Claude API to provide an intelligent explanation of
the gradient descent implementation. This serves as a tool to ensure proper
understanding of the algorithm used in this program.
'''
import anthropic
import os

def query_claude(prompt, api_key=None):
    # Try to get API key from environment variable
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    # Validate that we have an API key before proceeding
    if not api_key:
        raise ValueError("API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter")
    
    # Initialize the Anthropic client with the API key
    client = anthropic.Anthropic(api_key=api_key)

    # Create a message request to Claude
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract and return the text content from Claude's response
    return message.content[0].text

def response_claude():
    prompt = """
I'm implementing gradient descent for linear regression with the following setup:

Hypothesis: h(x) = θ₀ + θ₁·x
Cost function: J(θ) = (1/2m) Σ(h(xᵢ) - yᵢ)²

I derived these gradient formulas:
∂J/∂θ₀ = (1/m) Σ(h(xᵢ) - yᵢ)
∂J/∂θ₁ = (1/m) Σ(h(xᵢ) - yᵢ) · xᵢ

Please explain the algorithm I used in a concise, yet detailed manner.
"""
    response = query_claude(prompt)
    print(response)
    print("\n")
    return response