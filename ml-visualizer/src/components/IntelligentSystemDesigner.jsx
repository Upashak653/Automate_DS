import React, { useState } from 'react';
import { Brain, AlertTriangle, CheckCircle, TrendingUp, FileText, Download, Zap } from 'lucide-react';

const IntelligentSystemDesigner = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const analyzeSystem = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/api/isd/analyze-complete', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setReport(data.report);
    } catch (error) {
      console.error('Error:', error);
      alert('Analysis failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = () => {
    if (!report) return;
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'ml-architecture-report.json';
    a.click();
  };

  const getRiskColor = (score) => {
    if (score >= 70) return 'text-red-400';
    if (score >= 40) return 'text-orange-400';
    return 'text-green-400';
  };

  const getHealthColor = (score) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="min-h-screen bg-black p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-3 bg-gradient-to-br from-purple-600 to-blue-600 rounded-xl glow-purple shadow-lg shadow-purple-600/50">
              <Brain className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
                Intelligent System Designer
              </h1>
              <p className="text-gray-500 text-sm mt-1">
                AI-Powered ML Architecture Analysis • Enterprise Grade
              </p>
            </div>
          </div>
        </div>

        {/* File Upload */}
        <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl p-6 mb-6 hover:border-purple-600/50 transition-all shadow-xl">
          <h2 className="text-xl font-semibold mb-4 text-white">Upload Dataset</h2>
          <div className="flex gap-4">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="flex-1 p-3 bg-black border border-[#1a1a1a] rounded-lg text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-purple-600 file:to-blue-600 file:text-white hover:file:from-purple-700 hover:file:to-blue-700 cursor-pointer transition-all"
            />
            <button
              onClick={analyzeSystem}
              disabled={!file || loading}
              className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 disabled:from-gray-700 disabled:to-gray-800 disabled:cursor-not-allowed transition-all glow-purple shadow-lg shadow-purple-600/30"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  Analyzing...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Zap className="w-4 h-4" />
                  Analyze System
                </span>
              )}
            </button>
          </div>
        </div>

        {/* Report Display */}
        {report && (
          <>
            {/* Executive Summary */}
            <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-600/40 rounded-xl p-8 mb-6 shadow-2xl shadow-purple-900/20">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h2 className="text-3xl font-bold text-white mb-2">Executive Summary</h2>
                  <p className="text-gray-500 text-sm">AI-Generated Architecture Report</p>
                </div>
                <button
                  onClick={downloadReport}
                  className="flex items-center gap-2 px-5 py-2.5 bg-black border border-purple-600/50 text-purple-400 rounded-lg hover:bg-purple-600/10 hover:border-purple-600 transition-all shadow-lg shadow-purple-600/20"
                >
                  <Download className="w-4 h-4" />
                  Download
                </button>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-black border border-[#1a1a1a] rounded-xl p-5 hover:border-purple-600/70 hover:shadow-lg hover:shadow-purple-600/20 transition-all">
                  <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Project Viability</div>
                  <div className="text-3xl font-bold text-purple-400">{report.executive_summary.project_viability}</div>
                </div>
                <div className="bg-black border border-[#1a1a1a] rounded-xl p-5 hover:border-blue-600/70 hover:shadow-lg hover:shadow-blue-600/20 transition-all">
                  <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Data Health</div>
                  <div className="text-3xl font-bold text-blue-400">{report.executive_summary.data_health_score}/100</div>
                </div>
                <div className="bg-black border border-[#1a1a1a] rounded-xl p-5 hover:border-orange-600/70 hover:shadow-lg hover:shadow-orange-600/20 transition-all">
                  <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Risk Score</div>
                  <div className="text-3xl font-bold text-orange-400">{report.executive_summary.overall_risk_score}/100</div>
                </div>
                <div className="bg-black border border-[#1a1a1a] rounded-xl p-5 hover:border-green-600/70 hover:shadow-lg hover:shadow-green-600/20 transition-all">
                  <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Complexity</div>
                  <div className="text-3xl font-bold text-green-400 capitalize">{report.executive_summary.estimated_complexity}</div>
                </div>
              </div>

              <div className="mt-6 p-5 bg-black border border-purple-600/40 rounded-xl shadow-lg shadow-purple-600/10">
                <div className="font-semibold text-purple-400 mb-3 flex items-center gap-2">
                  <span className="text-2xl">💡</span>
                  <span>Key Recommendation</span>
                </div>
                <div className="text-gray-300 leading-relaxed">{report.executive_summary.key_recommendation}</div>
              </div>
            </div>

            {/* Tabs */}
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-xl mb-6 overflow-hidden shadow-2xl">
              <div className="flex border-b border-[#1a1a1a] overflow-x-auto bg-black">
                {['overview', 'data', 'model', 'risks', 'roadmap'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    className={`px-8 py-4 font-semibold capitalize transition-all whitespace-nowrap ${
                      activeTab === tab
                        ? 'bg-gradient-to-b from-purple-600/20 to-transparent text-purple-400 border-b-2 border-purple-600 shadow-lg shadow-purple-600/20'
                        : 'text-gray-500 hover:text-gray-300 hover:bg-[#0a0a0a]'
                    }`}
                  >
                    {tab}
                  </button>
                ))}
              </div>

              <div className="p-8">
                {/* Overview Tab */}
                {activeTab === 'overview' && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Problem Analysis</h3>
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-5 bg-black border border-[#1a1a1a] rounded-xl hover:border-purple-600/50 transition-all">
                          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Problem Type</div>
                          <div className="text-xl font-semibold text-white capitalize mt-1">
                            {report.problem_analysis.problem_classification.type}
                          </div>
                        </div>
                        <div className="p-5 bg-black border border-[#1a1a1a] rounded-xl hover:border-blue-600/50 transition-all">
                          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Subtype</div>
                          <div className="text-xl font-semibold text-white mt-1">
                            {report.problem_analysis.problem_classification.subtype.replace(/_/g, ' ')}
                          </div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-white mb-4">Recommended Model</h3>
                      <div className="p-8 bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-600/60 rounded-xl shadow-xl shadow-purple-600/20">
                        <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-3">
                          {report.executive_summary.recommended_model}
                        </div>
                        <div className="text-sm text-gray-400">
                          ✨ Best match for your dataset and problem type
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Data Tab */}
                {activeTab === 'data' && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-3 flex items-center gap-2">
                        <TrendingUp className="w-5 h-5 text-purple-400" />
                        Data Health Score: 
                        <span className={getHealthColor(report.data_diagnosis.health_score)}>
                          {report.data_diagnosis.health_score}/100
                        </span>
                      </h3>
                    </div>

                    {/* Red Flags */}
                    {report.data_diagnosis.red_flags.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-red-400 mb-3 flex items-center gap-2">
                          <AlertTriangle className="w-5 h-5" />
                          Critical Issues
                        </h4>
                        <div className="space-y-3">
                          {report.data_diagnosis.red_flags.map((flag, idx) => (
                            <div key={idx} className="p-4 bg-red-900/20 border border-red-600/50 rounded-lg">
                              <div className="font-semibold text-red-400">{flag.message}</div>
                              <div className="text-sm text-gray-400 mt-1">{flag.details}</div>
                              <div className="text-sm text-red-400 mt-2">Impact: {flag.impact}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Warnings */}
                    {report.data_diagnosis.warnings.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-yellow-400 mb-3">Warnings</h4>
                        <div className="space-y-3">
                          {report.data_diagnosis.warnings.map((warning, idx) => (
                            <div key={idx} className="p-4 bg-yellow-900/20 border border-yellow-600/50 rounded-lg">
                              <div className="font-semibold text-yellow-400">{warning.message}</div>
                              <div className="text-sm text-gray-400 mt-1">→ {warning.recommendation}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Repair Suggestions */}
                    {report.data_diagnosis.repair_suggestions.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-blue-400 mb-3">Repair Suggestions</h4>
                        <div className="space-y-3">
                          {report.data_diagnosis.repair_suggestions.map((suggestion, idx) => (
                            <div key={idx} className="p-4 bg-blue-900/20 border border-blue-600/50 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                <span className={`px-2 py-1 text-xs rounded font-semibold ${
                                  suggestion.priority === 'critical' ? 'bg-red-600 text-white' :
                                  suggestion.priority === 'high' ? 'bg-orange-600 text-white' :
                                  'bg-blue-600 text-white'
                                }`}>
                                  {suggestion.priority}
                                </span>
                                <span className="font-semibold text-blue-400">{suggestion.action}</span>
                              </div>
                              <div className="text-sm text-gray-400 mb-2">{suggestion.details}</div>
                              <code className="text-xs bg-[#0a0a0a] border border-[#2a2a2a] px-3 py-2 rounded block text-purple-400">
                                {suggestion.code_hint}
                              </code>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Model Tab */}
                {activeTab === 'model' && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Top Algorithm Recommendations</h3>
                      <div className="space-y-4">
                        {report.model_recommendations.top_algorithms.map((algo, idx) => (
                          <div key={idx} className="p-5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg hover:border-purple-600/50 transition-all">
                            <div className="flex justify-between items-start mb-3">
                              <div>
                                <div className="text-lg font-bold text-gray-100">{algo.name}</div>
                                <div className="text-sm text-gray-400">{algo.family}</div>
                              </div>
                              <div className="text-3xl font-bold text-purple-400">{algo.score}</div>
                            </div>
                            <div className="mb-3">
                              <div className="text-sm font-semibold text-gray-300 mb-2">Why this model:</div>
                              <ul className="text-sm text-gray-400 list-disc list-inside space-y-1">
                                {algo.reasoning.map((reason, i) => (
                                  <li key={i}>{reason}</li>
                                ))}
                              </ul>
                            </div>
                            <div className="text-sm text-gray-400">
                              <span className="font-semibold text-gray-300">Best for:</span> {algo.when_to_use}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Evaluation Metrics</h3>
                      <div className="grid grid-cols-2 gap-3">
                        {report.model_recommendations.evaluation_metrics.map((metric, idx) => (
                          <div key={idx} className={`p-4 rounded-lg ${metric.primary ? 'bg-purple-900/20 border border-purple-600/50' : 'bg-[#1a1a1a] border border-[#2a2a2a]'}`}>
                            <div className="font-semibold text-gray-100">{metric.name}</div>
                            <div className="text-sm text-gray-400 mt-1">{metric.reason}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Validation Strategy</h3>
                      <div className="p-5 bg-[#1a1a1a] border border-[#2a2a2a] rounded-lg">
                        <div className="font-semibold text-gray-100 mb-2">
                          {report.model_recommendations.validation_strategy.strategy}
                        </div>
                        <div className="text-sm text-gray-400 mb-3">
                          {report.model_recommendations.validation_strategy.reason}
                        </div>
                        <div className="text-sm text-purple-400">
                          Splits: {report.model_recommendations.validation_strategy.n_splits}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Risks Tab */}
                {activeTab === 'risks' && (
                  <div className="space-y-6">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="p-5 bg-red-900/20 border border-red-600/50 rounded-lg">
                        <div className="text-sm text-gray-400 mb-1">Overfitting Risk</div>
                        <div className={`text-3xl font-bold ${getRiskColor(report.risk_assessment.overfitting_risk.risk_score)}`}>
                          {report.risk_assessment.overfitting_risk.risk_level.toUpperCase()}
                        </div>
                        <div className="text-sm text-gray-400 mt-2">{report.risk_assessment.overfitting_risk.probability}</div>
                      </div>
                      <div className="p-5 bg-yellow-900/20 border border-yellow-600/50 rounded-lg">
                        <div className="text-sm text-gray-400 mb-1">Underfitting Risk</div>
                        <div className={`text-3xl font-bold ${getRiskColor(report.risk_assessment.underfitting_risk.risk_score)}`}>
                          {report.risk_assessment.underfitting_risk.risk_level.toUpperCase()}
                        </div>
                        <div className="text-sm text-gray-400 mt-2">{report.risk_assessment.underfitting_risk.probability}</div>
                      </div>
                    </div>

                    {report.risk_assessment.critical_warnings.length > 0 && (
                      <div>
                        <h4 className="font-semibold text-red-400 mb-3">Critical Warnings</h4>
                        <div className="space-y-3">
                          {report.risk_assessment.critical_warnings.map((warning, idx) => (
                            <div key={idx} className="p-4 bg-red-900/20 border border-red-600/50 rounded-lg">
                              <div className="font-semibold text-red-400">{warning.message}</div>
                              <div className="text-sm text-gray-400 mt-1">Type: {warning.type}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div>
                      <h4 className="font-semibold text-green-400 mb-3">Preventive Actions</h4>
                      <div className="space-y-3">
                        {report.risk_assessment.preventive_actions.map((action, idx) => (
                          <div key={idx} className="p-4 bg-green-900/20 border border-green-600/50 rounded-lg flex items-center gap-3">
                            <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0" />
                            <div className="flex-1">
                              <div className="font-semibold text-green-400">{action.action}</div>
                              <div className="text-xs text-gray-400 mt-1">Priority: {action.priority}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Roadmap Tab */}
                {activeTab === 'roadmap' && (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Implementation Roadmap</h3>
                      <div className="text-sm text-gray-400 mb-6">
                        Total Duration: {report.implementation_roadmap.total_estimated_duration}
                      </div>
                      <div className="space-y-4">
                        {report.implementation_roadmap.phases.map((phase, idx) => (
                          <div key={idx} className="p-5 border-l-4 border-purple-600 bg-[#1a1a1a] rounded-lg">
                            <div className="flex justify-between items-start mb-3">
                              <div>
                                <div className="text-lg font-bold text-gray-100">Phase {phase.phase}: {phase.name}</div>
                                <div className="text-sm text-gray-400">{phase.duration}</div>
                              </div>
                            </div>
                            <div className="mb-3">
                              <div className="text-sm font-semibold text-gray-300 mb-2">Tasks:</div>
                              <ul className="text-sm text-gray-400 list-disc list-inside space-y-1">
                                {phase.tasks.map((task, i) => (
                                  <li key={i}>{task}</li>
                                ))}
                              </ul>
                            </div>
                            <div className="text-sm text-purple-400">
                              <span className="font-semibold">Deliverable:</span> {phase.deliverable}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h3 className="text-xl font-semibold text-gray-100 mb-4">Success Metrics</h3>
                      <div className="p-5 bg-green-900/20 border border-green-600/50 rounded-lg">
                        <div className="mb-3">
                          <span className="font-semibold text-gray-100">Primary Metric:</span> 
                          <span className="text-green-400 ml-2">{report.success_metrics.primary_metric}</span>
                        </div>
                        <div className="mb-3">
                          <span className="font-semibold text-gray-100">Target Value:</span> 
                          <span className="text-green-400 ml-2">{report.success_metrics.target_value}</span>
                        </div>
                        <div>
                          <span className="font-semibold text-gray-100">Minimum Acceptable:</span> 
                          <span className="text-yellow-400 ml-2">{report.success_metrics.minimum_acceptable}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default IntelligentSystemDesigner;
