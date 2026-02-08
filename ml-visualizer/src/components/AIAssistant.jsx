import { useState, useRef, useEffect } from 'react';
import { MessageSquare, Send, X, Settings, Eye, Loader, Sparkles } from 'lucide-react';

export default function AIAssistant({ analysis }) {
  const [isOpen, setIsOpen] = useState(false);
  const [apiKey, setApiKey] = useState(localStorage.getItem('openai_api_key') || '');
  const [showSettings, setShowSettings] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [canSeeScreen, setCanSeeScreen] = useState(true);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const saveApiKey = () => {
    localStorage.setItem('openai_api_key', apiKey);
    setShowSettings(false);
    if (messages.length === 0) {
      addSystemMessage('API key saved! I can now assist you with your ML analysis. Ask me anything about your data!');
    }
  };

  const addSystemMessage = (content) => {
    setMessages(prev => [...prev, { role: 'assistant', content, timestamp: new Date() }]);
  };

  const generateDataContext = () => {
    if (!analysis) return '';
    
    const context = {
      dataShape: analysis.dataShape,
      problemType: analysis.problemType,
      targetColumn: analysis.targetColumn,
      validation: analysis.validation,
      columnSummary: analysis.columnAnalysis.map(col => ({
        name: col.name,
        type: col.type,
        uniqueCount: col.uniqueCount,
        missingPercent: col.missingPercent,
        ...(col.type === 'numeric' && { mean: col.mean, median: col.median, min: col.min, max: col.max })
      })),
      topRecommendation: analysis.recommendations[0],
      insights: analysis.insights
    };

    return `
Current Dataset Analysis:
- Problem Type: ${context.problemType}
- Target Variable: ${context.targetColumn}
- Dataset Size: ${context.dataShape.rows} rows × ${context.dataShape.columns} columns
- Numeric Features: ${context.dataShape.numericFeatures}
- Categorical Features: ${context.dataShape.categoricalFeatures}
- Recommended Model: ${context.topRecommendation.name} (${context.topRecommendation.confidence}% confidence)

Key Insights:
${context.insights.map(i => `- ${i.message}: ${i.detail}`).join('\n')}

Column Details:
${context.columnSummary.map(col => 
  `- ${col.name} (${col.type}): ${col.uniqueCount} unique values, ${col.missingPercent}% missing`
).join('\n')}
`;
  };

  const sendMessage = async () => {
    if (!input.trim() || !apiKey) {
      if (!apiKey) {
        addSystemMessage('Please set your OpenAI API key in settings first.');
      }
      return;
    }

    const userMessage = { role: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const systemPrompt = `You are an expert ML engineer assistant helping analyze a dataset. ${
        canSeeScreen ? `Here's the current data analysis:\n${generateDataContext()}` : ''
      }

Provide concise, actionable advice about:
- Data preprocessing steps
- Feature engineering suggestions
- Model selection rationale
- Handling data quality issues
- Best practices for the specific problem

Keep responses focused and practical. Use bullet points for clarity.`;

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: systemPrompt },
            ...messages.slice(-6).map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: input }
          ],
          temperature: 0.7,
          max_tokens: 800
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error?.message || 'API request failed');
      }

      const data = await response.json();
      const assistantMessage = {
        role: 'assistant',
        content: data.choices[0].message.content,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      addSystemMessage(`Error: ${error.message}. Please check your API key and try again.`);
    } finally {
      setIsLoading(false);
    }
  };

  const quickQuestions = [
    "What preprocessing steps should I take?",
    "Why was this model recommended?",
    "How should I handle missing data?",
    "What features are most important?",
    "How do I handle class imbalance?",
    "Should I remove outliers?"
  ];

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-4 rounded-full shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-110 z-50 flex items-center gap-2"
      >
        <Sparkles className="w-6 h-6" />
        <span className="font-semibold">AI Assistant</span>
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] bg-white rounded-xl shadow-2xl flex flex-col z-50 border border-gray-200">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-600 to-indigo-600 text-white p-4 rounded-t-xl flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5" />
              <h3 className="font-bold">ML Assistant</h3>
              {canSeeScreen && (
                <span className="text-xs bg-white/20 px-2 py-1 rounded-full flex items-center gap-1">
                  <Eye className="w-3 h-3" />
                  Viewing Data
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="hover:bg-white/20 p-1 rounded transition-colors"
              >
                <Settings className="w-5 h-5" />
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="hover:bg-white/20 p-1 rounded transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <div className="p-4 bg-gray-50 border-b border-gray-200">
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                OpenAI API Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
              <div className="flex items-center gap-2 mt-3">
                <button
                  onClick={saveApiKey}
                  className="flex-1 bg-purple-600 text-white px-3 py-2 rounded-lg text-sm font-semibold hover:bg-purple-700 transition-colors"
                >
                  Save Key
                </button>
                <button
                  onClick={() => setCanSeeScreen(!canSeeScreen)}
                  className={`flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-semibold transition-colors ${
                    canSeeScreen ? 'bg-green-100 text-green-700' : 'bg-gray-200 text-gray-700'
                  }`}
                >
                  <Eye className="w-4 h-4" />
                  {canSeeScreen ? 'On' : 'Off'}
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="text-purple-600 hover:underline">OpenAI Platform</a>
              </p>
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-center py-8">
                <Sparkles className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                <p className="text-gray-600 font-semibold mb-2">Welcome to ML Assistant!</p>
                <p className="text-sm text-gray-500 mb-4">
                  {apiKey ? 'Ask me anything about your data analysis' : 'Set your API key to get started'}
                </p>
                {apiKey && analysis && (
                  <div className="space-y-2">
                    <p className="text-xs text-gray-500 font-semibold">Quick Questions:</p>
                    {quickQuestions.slice(0, 3).map((q, idx) => (
                      <button
                        key={idx}
                        onClick={() => setInput(q)}
                        className="block w-full text-left text-xs bg-purple-50 hover:bg-purple-100 text-purple-700 px-3 py-2 rounded-lg transition-colors"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 ${
                    msg.role === 'user'
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  <p className="text-xs opacity-70 mt-1">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </p>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center gap-2">
                  <Loader className="w-4 h-4 animate-spin text-purple-600" />
                  <span className="text-sm text-gray-600">Thinking...</span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Questions */}
          {messages.length > 0 && !isLoading && (
            <div className="px-4 py-2 border-t border-gray-200 bg-gray-50">
              <div className="flex gap-2 overflow-x-auto pb-2">
                {quickQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => setInput(q)}
                    className="text-xs bg-white hover:bg-purple-50 text-purple-700 px-3 py-1 rounded-full border border-purple-200 whitespace-nowrap transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                placeholder={apiKey ? "Ask about your data..." : "Set API key first..."}
                disabled={!apiKey || isLoading}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed"
              />
              <button
                onClick={sendMessage}
                disabled={!apiKey || !input.trim() || isLoading}
                className="bg-purple-600 text-white p-2 rounded-lg hover:bg-purple-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
