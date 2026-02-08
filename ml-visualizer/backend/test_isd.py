"""
Test script for ISD modules
Run this to verify all ISD modules are working correctly
"""

import pandas as pd
import numpy as np
from isd_data_intelligence import DataIntelligenceEngine
from isd_problem_understanding import ProblemUnderstandingModule
from isd_model_architect import ModelArchitectModule
from isd_failure_predictor import FailurePredictionModule
from isd_report_generator import ReportGenerator

def create_test_dataset():
    """Create a simple test dataset"""
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples) * 2,
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.randn(n_samples) + 5,
        'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Imbalanced
    })
    
    # Add some missing values
    df.loc[np.random.choice(n_samples, 50, replace=False), 'feature1'] = np.nan
    
    # Add some outliers
    df.loc[np.random.choice(n_samples, 10, replace=False), 'feature2'] = 100
    
    return df

def test_data_intelligence():
    """Test Data Intelligence Engine"""
    print("\n" + "="*60)
    print("Testing Data Intelligence Engine")
    print("="*60)
    
    df = create_test_dataset()
    engine = DataIntelligenceEngine(df)
    analysis = engine.analyze()
    
    print(f"✓ Health Score: {analysis['health_score']}/100")
    print(f"✓ Red Flags: {len(analysis['red_flags'])}")
    print(f"✓ Warnings: {len(analysis['warnings'])}")
    print(f"✓ Repair Suggestions: {len(analysis['repair_suggestions'])}")
    
    if analysis['health_score'] > 0:
        print("✅ Data Intelligence Engine: PASSED")
        return True
    else:
        print("❌ Data Intelligence Engine: FAILED")
        return False

def test_problem_understanding():
    """Test Problem Understanding Module"""
    print("\n" + "="*60)
    print("Testing Problem Understanding Module")
    print("="*60)
    
    df = create_test_dataset()
    module = ProblemUnderstandingModule(df)
    analysis = module.analyze()
    
    print(f"✓ Problem Type: {analysis['problem_type']}")
    print(f"✓ Problem Subtype: {analysis['problem_subtype']}")
    print(f"✓ Complexity: {analysis['task_complexity']['complexity_level']}")
    print(f"✓ Risk Level: {analysis['risk_profile']['risk_level']}")
    
    if analysis['problem_type'] in ['classification', 'regression']:
        print("✅ Problem Understanding Module: PASSED")
        return True
    else:
        print("❌ Problem Understanding Module: FAILED")
        return False

def test_model_architect():
    """Test Model Architect Module"""
    print("\n" + "="*60)
    print("Testing Model Architect Module")
    print("="*60)
    
    df = create_test_dataset()
    
    # Need both analyses
    data_engine = DataIntelligenceEngine(df)
    data_analysis = data_engine.analyze()
    
    problem_module = ProblemUnderstandingModule(df)
    problem_analysis = problem_module.analyze()
    
    architect = ModelArchitectModule(problem_analysis, data_analysis)
    recommendations = architect.recommend()
    
    print(f"✓ Top Algorithm: {recommendations['algorithm_recommendations'][0]['name']}")
    print(f"✓ Loss Function: {recommendations['loss_function']['name']}")
    print(f"✓ Validation Strategy: {recommendations['validation_strategy']['strategy']}")
    print(f"✓ Preprocessing Steps: {len(recommendations['preprocessing_pipeline'])}")
    
    if len(recommendations['algorithm_recommendations']) > 0:
        print("✅ Model Architect Module: PASSED")
        return True
    else:
        print("❌ Model Architect Module: FAILED")
        return False

def test_failure_predictor():
    """Test Failure Prediction Module"""
    print("\n" + "="*60)
    print("Testing Failure Prediction Module")
    print("="*60)
    
    df = create_test_dataset()
    
    # Need both analyses
    data_engine = DataIntelligenceEngine(df)
    data_analysis = data_engine.analyze()
    
    problem_module = ProblemUnderstandingModule(df)
    problem_analysis = problem_module.analyze()
    
    predictor = FailurePredictionModule(problem_analysis, data_analysis)
    predictions = predictor.predict_failures()
    
    print(f"✓ Overfitting Risk: {predictions['overfitting_risk']['risk_level']}")
    print(f"✓ Underfitting Risk: {predictions['underfitting_risk']['risk_level']}")
    print(f"✓ Data Sufficiency: {predictions['data_sufficiency']['status']}")
    print(f"✓ Overall Risk Score: {predictions['overall_risk_score']}/100")
    
    if predictions['overall_risk_score'] >= 0:
        print("✅ Failure Prediction Module: PASSED")
        return True
    else:
        print("❌ Failure Prediction Module: FAILED")
        return False

def test_report_generator():
    """Test Report Generator"""
    print("\n" + "="*60)
    print("Testing Report Generator")
    print("="*60)
    
    df = create_test_dataset()
    
    # Run all analyses
    data_engine = DataIntelligenceEngine(df)
    data_analysis = data_engine.analyze()
    
    problem_module = ProblemUnderstandingModule(df)
    problem_analysis = problem_module.analyze()
    
    model_architect = ModelArchitectModule(problem_analysis, data_analysis)
    model_recommendations = model_architect.recommend()
    
    failure_predictor = FailurePredictionModule(problem_analysis, data_analysis)
    failure_predictions = failure_predictor.predict_failures()
    
    # Generate report
    report_gen = ReportGenerator(
        data_analysis, problem_analysis,
        model_recommendations, failure_predictions
    )
    report = report_gen.generate_report()
    
    print(f"✓ Report Sections: {len(report)}")
    print(f"✓ Project Viability: {report['executive_summary']['project_viability']}")
    print(f"✓ Recommended Model: {report['executive_summary']['recommended_model']}")
    print(f"✓ Roadmap Phases: {len(report['implementation_roadmap']['phases'])}")
    
    if 'executive_summary' in report:
        print("✅ Report Generator: PASSED")
        return True
    else:
        print("❌ Report Generator: FAILED")
        return False

def test_complete_pipeline():
    """Test complete ISD pipeline"""
    print("\n" + "="*60)
    print("Testing Complete ISD Pipeline")
    print("="*60)
    
    df = create_test_dataset()
    
    # Run complete pipeline
    data_engine = DataIntelligenceEngine(df)
    data_analysis = data_engine.analyze()
    
    problem_module = ProblemUnderstandingModule(df)
    problem_analysis = problem_module.analyze()
    
    model_architect = ModelArchitectModule(problem_analysis, data_analysis)
    model_recommendations = model_architect.recommend()
    
    failure_predictor = FailurePredictionModule(problem_analysis, data_analysis)
    failure_predictions = failure_predictor.predict_failures()
    
    report_gen = ReportGenerator(
        data_analysis, problem_analysis,
        model_recommendations, failure_predictions
    )
    report = report_gen.generate_report()
    
    print("\n📊 COMPLETE ANALYSIS RESULTS:")
    print(f"  • Data Health: {report['executive_summary']['data_health_score']}/100")
    print(f"  • Problem Type: {report['executive_summary']['problem_type']}")
    print(f"  • Recommended Model: {report['executive_summary']['recommended_model']}")
    print(f"  • Risk Score: {report['executive_summary']['overall_risk_score']}/100")
    print(f"  • Project Viability: {report['executive_summary']['project_viability']}")
    
    print("\n✅ Complete ISD Pipeline: PASSED")
    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ISD MODULE TEST SUITE")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Data Intelligence", test_data_intelligence()))
    except Exception as e:
        print(f"❌ Data Intelligence: FAILED - {e}")
        results.append(("Data Intelligence", False))
    
    try:
        results.append(("Problem Understanding", test_problem_understanding()))
    except Exception as e:
        print(f"❌ Problem Understanding: FAILED - {e}")
        results.append(("Problem Understanding", False))
    
    try:
        results.append(("Model Architect", test_model_architect()))
    except Exception as e:
        print(f"❌ Model Architect: FAILED - {e}")
        results.append(("Model Architect", False))
    
    try:
        results.append(("Failure Predictor", test_failure_predictor()))
    except Exception as e:
        print(f"❌ Failure Predictor: FAILED - {e}")
        results.append(("Failure Predictor", False))
    
    try:
        results.append(("Report Generator", test_report_generator()))
    except Exception as e:
        print(f"❌ Report Generator: FAILED - {e}")
        results.append(("Report Generator", False))
    
    try:
        results.append(("Complete Pipeline", test_complete_pipeline()))
    except Exception as e:
        print(f"❌ Complete Pipeline: FAILED - {e}")
        results.append(("Complete Pipeline", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<40} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! ISD is ready to use!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
