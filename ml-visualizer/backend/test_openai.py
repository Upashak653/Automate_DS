"""
Quick test script for OpenAI API integration
"""

from ai_overfitting_agent import AIOverfittingAgent

# Test data
model_results = {
    'model_type': 'RandomForestClassifier',
    'train_accuracy': 0.95,
    'accuracy': 0.75,
    'train_size': 800,
    'test_size': 200,
    'features': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
    'cv_mean': 0.78,
    'cv_std': 0.05
}

dataset_info = {
    'shape': (1000, 6),
    'columns': ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'target']
}

# Test with your API key
api_key = input("Enter your OpenAI API key: ")

print("\n🤖 Testing AI Overfitting Agent...\n")

agent = AIOverfittingAgent(api_key=api_key)
result = agent.analyze_overfitting(model_results, dataset_info)

if 'error' in result:
    print(f"❌ Error: {result['message']}")
    print(f"Details: {result['error']}")
    if 'fallback_recommendations' in result:
        print("\n📋 Fallback Recommendations:")
        for rec in result['fallback_recommendations']:
            print(f"  • {rec}")
else:
    print(f"✅ Success!")
    print(f"\n📊 Overfitting Score: {result['overfitting_score']}/100")
    print(f"Severity: {result['overfitting_severity']}")
    print(f"Detected: {result['overfitting_detected']}")
    print(f"\n🤖 AI Analysis:\n{result['ai_analysis'][:500]}...")
    print(f"\n⚡ Quick Fixes: {len(result['quick_fixes'])} available")

print("\n✅ Test complete!")
