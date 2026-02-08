"""
ISD API Integration
Adds Intelligent System Designer endpoints to Flask app
"""

from flask import request, jsonify
import pandas as pd
import sys

# Import ISD modules
from isd_data_intelligence import DataIntelligenceEngine
from isd_problem_understanding import ProblemUnderstandingModule
from isd_model_architect import ModelArchitectModule
from isd_failure_predictor import FailurePredictionModule
from isd_report_generator import ReportGenerator


def read_csv_safe(file):
    """Safely read CSV file"""
    try:
        return pd.read_csv(file)
    except pd.errors.ParserError:
        file.seek(0)
        try:
            return pd.read_csv(file, on_bad_lines='skip')
        except:
            pass
    except UnicodeDecodeError:
        file.seek(0)
        try:
            return pd.read_csv(file, encoding='latin-1')
        except:
            pass
    file.seek(0)
    return pd.read_csv(file, sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')


def register_isd_routes(app):
    """Register ISD routes with Flask app"""
    
    @app.route('/api/isd/analyze-complete', methods=['POST'])
    def isd_analyze_complete():
        """
        Complete ISD analysis - all modules in one call
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
            print(f"ISD: Analyzing dataset with shape {df.shape}")
            
            # Run all ISD modules
            print("ISD: Running Data Intelligence Engine...")
            data_engine = DataIntelligenceEngine(df)
            data_analysis = data_engine.analyze()
            
            print("ISD: Running Problem Understanding Module...")
            problem_module = ProblemUnderstandingModule(df)
            problem_analysis = problem_module.analyze()
            
            print("ISD: Running Model Architect Module...")
            model_architect = ModelArchitectModule(problem_analysis, data_analysis)
            model_recommendations = model_architect.recommend()
            
            print("ISD: Running Failure Predictor Module...")
            failure_predictor = FailurePredictionModule(problem_analysis, data_analysis)
            failure_predictions = failure_predictor.predict_failures()
            
            print("ISD: Generating Report...")
            # Generate report
            report_gen = ReportGenerator(
                data_analysis, problem_analysis,
                model_recommendations, failure_predictions
            )
            full_report = report_gen.generate_report()
            
            print("ISD: Analysis complete!")
            
            return jsonify({
                'success': True,
                'report': full_report,
                'data_intelligence': data_analysis,
                'problem_understanding': problem_analysis,
                'model_recommendations': model_recommendations,
                'failure_predictions': failure_predictions
            })
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ISD Error: {error_details}")
            return jsonify({'error': str(e), 'details': error_details}), 400
    
    @app.route('/api/isd/data-intelligence', methods=['POST'])
    def isd_data_intelligence():
        """
        Data Intelligence Engine - analyze dataset health
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
            engine = DataIntelligenceEngine(df)
            analysis = engine.analyze()
            
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/isd/problem-understanding', methods=['POST'])
    def isd_problem_understanding():
        """
        Problem Understanding Module - classify ML problem
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
            module = ProblemUnderstandingModule(df)
            analysis = module.analyze()
            
            return jsonify({
                'success': True,
                'analysis': analysis
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/isd/model-architect', methods=['POST'])
    def isd_model_architect():
        """
        Model Architect Module - recommend ML architecture
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
            # Need both data and problem analysis
            data_engine = DataIntelligenceEngine(df)
            data_analysis = data_engine.analyze()
            
            problem_module = ProblemUnderstandingModule(df)
            problem_analysis = problem_module.analyze()
            
            architect = ModelArchitectModule(problem_analysis, data_analysis)
            recommendations = architect.recommend()
            
            return jsonify({
                'success': True,
                'recommendations': recommendations
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/isd/failure-prediction', methods=['POST'])
    def isd_failure_prediction():
        """
        Failure Prediction Module - predict potential failures
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
            # Need both analyses
            data_engine = DataIntelligenceEngine(df)
            data_analysis = data_engine.analyze()
            
            problem_module = ProblemUnderstandingModule(df)
            problem_analysis = problem_module.analyze()
            
            predictor = FailurePredictionModule(problem_analysis, data_analysis)
            predictions = predictor.predict_failures()
            
            return jsonify({
                'success': True,
                'predictions': predictions
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/isd/generate-report', methods=['POST'])
    def isd_generate_report():
        """
        Generate comprehensive ML architecture report
        """
        try:
            file = request.files['file']
            df = read_csv_safe(file)
            
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
            summary_text = report_gen.export_summary()
            json_export = report_gen.export_json()
            
            return jsonify({
                'success': True,
                'report': report,
                'summary_text': summary_text,
                'json_export': json_export
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    @app.route('/api/isd/health-check', methods=['GET'])
    def isd_health_check():
        """
        Health check for ISD system
        """
        # Test imports
        try:
            from isd_data_intelligence import DataIntelligenceEngine
            from isd_problem_understanding import ProblemUnderstandingModule
            from isd_model_architect import ModelArchitectModule
            from isd_failure_predictor import FailurePredictionModule
            from isd_report_generator import ReportGenerator
            
            return jsonify({
                'status': 'operational',
                'modules': [
                    'Data Intelligence Engine',
                    'Problem Understanding Module',
                    'Model Architect Module',
                    'Failure Prediction Module',
                    'Report Generator'
                ],
                'version': '1.0.0',
                'imports': 'success'
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'error': str(e),
                'version': '1.0.0'
            }), 500
