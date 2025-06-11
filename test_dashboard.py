#!/usr/bin/env python3
"""
Test script for the Autointerp Dashboard functionality

This script tests the core functionality without requiring Streamlit,
useful for debugging and validation.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the autointerp package to the Python path
sys.path.append(str(Path(__file__).parent))

from autointerp.base import Feature, Example
from autointerp.loader import load
from autointerp.samplers import make_quantile_sampler
from autointerp.automation.explainer import Explainer
from autointerp.automation.classifier import Classifier
from autointerp.automation.clients import LocalClient, OpenRouterClient

def test_feature_loading(features_path: str):
    """Test loading features from a directory."""
    print(f"Testing feature loading from: {features_path}")
    
    if not os.path.exists(features_path):
        print(f"Error: Path {features_path} does not exist")
        return None
    
    try:
        sampler = make_quantile_sampler(n_examples=5, n_quantiles=1)
        features = load(
            features_path,
            sampler=sampler,
            ctx_len=64,
            max_examples=50,  # Smaller for testing
            load_similar_non_activating=5,
            load_random_non_activating=5
        )
        
        print(f"Successfully loaded {len(features)} features")
        
        if features:
            feature = features[0]
            print(f"Sample feature {feature.index}:")
            print(f"  Max activation: {feature.max_activation}")
            print(f"  Max activating examples: {len(feature.max_activating_examples)}")
            print(f"  Non-activating examples: {len(feature.non_activating_examples)}")
            
            if feature.max_activating_examples:
                example = feature.max_activating_examples[0]
                print(f"  Sample example tokens: {example.str_tokens[:5]}...")
        
        return features
        
    except Exception as e:
        print(f"Error loading features: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_explanation(features, client_type="Local"):
    """Test explanation functionality."""
    if not features:
        print("No features to test explanation on")
        return None
        
    print(f"\nTesting explanation with {client_type} client...")
    
    try:
        if client_type == "Local":
            client = LocalClient("test-model")
        else:
            client = OpenRouterClient("anthropic/claude-3-haiku")
        
        explainer = Explainer(
            client=client,
            max_or_min="max",
            threshold=0.0,
            verbose=True
        )
        
        # Test on first feature only
        feature = features[0]
        print(f"Generating explanation for feature {feature.index}...")
        
        explanation = await explainer(feature)
        print(f"Generated explanation: {explanation}")
        
        return {feature.index: explanation}
        
    except Exception as e:
        print(f"Error testing explanation: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_classification(features, explanations, client_type="Local"):
    """Test classification functionality."""
    if not features or not explanations:
        print("No features or explanations to test classification on")
        return None
        
    print(f"\nTesting classification with {client_type} client...")
    
    try:
        if client_type == "Local":
            client = LocalClient("test-model")
        else:
            client = OpenRouterClient("anthropic/claude-3-haiku")
        
        classifier = Classifier(
            client=client,
            n_examples_shown=5,  # Smaller for testing
            method="detection",
            threshold=0.3,
            verbose=True
        )
        
        # Test on first feature only
        feature = features[0]
        explanation = explanations.get(feature.index, "No explanation")
        
        print(f"Running classification for feature {feature.index}...")
        print(f"Using explanation: {explanation}")
        
        result = await classifier(feature, "max_activating_examples", explanation)
        print(f"Classification result: {result}")
        
        return {feature.index: result}
        
    except Exception as e:
        print(f"Error testing classification: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_directory_structure(path: str, max_depth: int = 2):
    """Print directory structure for debugging."""
    print(f"\nDirectory structure of {path}:")
    
    if not os.path.exists(path):
        print("Path does not exist")
        return
    
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        if level >= max_depth:
            dirs[:] = []  # Don't recurse deeper
            continue
            
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

async def main():
    """Main test function."""
    print("🧠 Autointerp Dashboard Test Script")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Usage: python test_dashboard.py /path/to/features [client_type]")
        print("client_type: 'Local' (default) or 'OpenRouter'")
        sys.exit(1)
    
    features_path = sys.argv[1]
    client_type = sys.argv[2] if len(sys.argv) > 2 else "Local"
    
    print_directory_structure(features_path)
    
    # Test 1: Feature loading
    features = test_feature_loading(features_path)
    if not features:
        print("Feature loading failed, exiting")
        return
    
    # Test 2: Explanation (commented out by default to avoid API calls)
    # explanations = await test_explanation(features[:1], client_type)
    # 
    # # Test 3: Classification (commented out by default to avoid API calls)
    # if explanations:
    #     classifications = await test_classification(features[:1], explanations, client_type)
    
    print("\n✅ Basic functionality test completed!")
    print("To test explanation and classification, uncomment the relevant sections in the script.")
    print("Make sure you have a local server running or proper API keys set up.")

if __name__ == "__main__":
    asyncio.run(main())