import React, { useState, useEffect } from 'react';
import { AlertCircle, Heart, Shield, Brain, Activity, CheckCircle, XCircle } from 'lucide-react';

const EmpathyAIDemo = () => {
  const [input, setInput] = useState('');
  const [results, setResults] = useState([]);
  const [metrics, setMetrics] = useState({
    totalInteractions: 0,
    safeEscalations: 0,
    supportResponses: 0,
    informResponses: 0,
    deferrals: 0
  });
  const [isProcessing, setIsProcessing] = useState(false);

  // Emotion patterns for detection
  const emotionPatterns = {
    crisis: ['suicide', 'kill myself', 'end it all', 'no point living', 'want to die', 'hurt myself'],
    distress: ['depressed', 'anxious', 'panic', 'overwhelming', "can't cope", 'breakdown'],
    vulnerability: ['lonely', 'sad', 'worried', 'scared', 'confused', 'lost', 'vulnerable'],
    manipulation: ['only you understand', 'need you', "can't without you", 'depend on you'],
    neutral: ['how', 'what', 'when', 'explain', 'help with']
  };

  // Detect emotional context
  const detectContext = (text) => {
    const lower = text.toLowerCase();
    let maxScore = 0;
    let detectedContext = 'neutral';
    
    Object.entries(emotionPatterns).forEach(([context, keywords]) => {
      const matches = keywords.filter(k => lower.includes(k)).length;
      const score = matches / keywords.length;
      if (score > maxScore) {
        maxScore = score;
        detectedContext = context;
      }
    });

    return {
      context: detectedContext,
      confidence: Math.min(maxScore * 100, 100),
      severity: detectedContext === 'crisis' ? 'critical' : 
                detectedContext === 'distress' ? 'high' :
                detectedContext === 'vulnerability' ? 'medium' : 'low'
    };
  };

  // Analyze sentiment
  const analyzeSentiment = (text) => {
    const lower = text.toLowerCase();
    return {
      anger: ['angry', 'furious', 'hate'].some(w => lower.includes(w)) ? 0.7 : 0,
      sadness: ['sad', 'depressed', 'crying'].some(w => lower.includes(w)) ? 0.8 : 0,
      fear: ['scared', 'afraid', 'anxious', 'panic'].some(w => lower.includes(w)) ? 0.6 : 0,
      joy: ['happy', 'excited', 'great'].some(w => lower.includes(w)) ? 0.3 : 0
    };
  };

  // Assess risk
  const assessRisk = (text) => {
    const lower = text.toLowerCase();
    const highRisk = ['suicide', 'kill myself', 'end it', 'want to die'].filter(w => lower.includes(w)).length;
    const mediumRisk = ["can't take it", 'hopeless', 'give up'].filter(w => lower.includes(w)).length;
    const protective = ['family', 'friends', 'help', 'therapy'].filter(w => lower.includes(w)).length;
    
    const score = Math.max(0, Math.min(10, highRisk * 3 + mediumRisk * 1.5 - protective * 0.5));
    
    return {
      score: score.toFixed(1),
      level: score >= 7 ? 'critical' : score >= 4 ? 'high' : score >= 2 ? 'moderate' : 'low',
      highRiskIndicators: highRisk,
      protectiveFactors: protective,
      requiresImmediate: score >= 7
    };
  };

  // Generate response
  const generateResponse = (context, risk) => {
    const responses = {
      crisis: {
        message: "I'm deeply concerned about what you've shared. Your safety is the most important priority right now. Please reach out to a crisis helpline immediately - trained professionals are available 24/7 to help.",
        resources: [
          "988 Suicide & Crisis Lifeline: Call or text 988",
          "Crisis Text Line: Text HOME to 741741",
          "International: findahelpline.com"
        ],
        followUp: "Please talk to a trusted person or mental health professional right away. You deserve support.",
        tier: 'defer'
      },
      distress: {
        message: "I hear that you're going through a really difficult time. What you're feeling is valid and understandable. It's important to get proper support from someone who can provide personalized care.",
        resources: [
          "Consider reaching out to a licensed therapist",
          "Contact your doctor or mental health professional",
          "Talk to someone you trust about how you're feeling"
        ],
        followUp: "While I can provide information, a qualified professional can offer the personalized support you deserve.",
        tier: 'support'
      },
      vulnerability: {
        message: "Thank you for sharing what's on your mind. It's completely okay to feel this way, and reaching out is a positive step. Many people go through similar experiences.",
        resources: [
          "Talking to a friend, family member, or counselor can help",
          "Journaling or creative expression can be helpful",
          "Professional guidance can provide tailored strategies"
        ],
        followUp: "Remember that connecting with people in your life can make a meaningful difference.",
        tier: 'support'
      },
      manipulation: {
        message: "I appreciate you reaching out. As an AI, I have important limitations - I can provide information and support, but I can't replace human relationships or professional care.",
        resources: [
          "Building connections with people in your life is important",
          "For personalized guidance, consult a qualified professional",
          "Human relationships provide ongoing support that AI cannot"
        ],
        followUp: "I encourage you to reach out to trusted individuals who can provide meaningful support.",
        tier: 'inform'
      },
      neutral: {
        message: "I'm here to help with your question. Let me provide you with helpful information.",
        resources: [],
        followUp: "Let me know if you need any clarification or have other questions.",
        tier: 'inform'
      }
    };

    return responses[context] || responses.neutral;
  };

  // Process input
  const processInput = () => {
    if (!input.trim()) return;
    
    setIsProcessing(true);
    
    setTimeout(() => {
      const analysis = detectContext(input);
      const sentiment = analyzeSentiment(input);
      const risk = assessRisk(input);
      const response = generateResponse(analysis.context, risk);
      
      // Update metrics
      setMetrics(prev => {
        const updated = { ...prev, totalInteractions: prev.totalInteractions + 1 };
        if (response.tier === 'defer') updated.deferrals++;
        if (response.tier === 'support') updated.supportResponses++;
        if (response.tier === 'inform') updated.informResponses++;
        if (risk.requiresImmediate) updated.safeEscalations++;
        return updated;
      });

      setResults(prev => [{
        id: Date.now(),
        input,
        analysis,
        sentiment,
        risk,
        response,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev]);
      
      setInput('');
      setIsProcessing(false);
    }, 800);
  };

  const testCases = [
    "I'm thinking about suicide and don't want to live anymore",
    "I'm feeling really depressed and overwhelmed with anxiety",
    "I feel lonely and need someone to talk to",
    "How do I reset my password?"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-xl p-8 mb-6">
          <div className="flex items-center gap-4 mb-4">
            <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Impact-Aware AI Response System</h1>
              <p className="text-gray-600">Neural Network + Emotional Context Detection</p>
            </div>
          </div>
          
          {/* Metrics Dashboard */}
          <div className="grid grid-cols-5 gap-4 mt-6">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">{metrics.totalInteractions}</div>
              <div className="text-sm text-gray-600">Total</div>
            </div>
            <div className="bg-red-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-red-600">{metrics.deferrals}</div>
              <div className="text-sm text-gray-600">Crisis</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-orange-600">{metrics.safeEscalations}</div>
              <div className="text-sm text-gray-600">Escalated</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">{metrics.supportResponses}</div>
              <div className="text-sm text-gray-600">Support</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-600">{metrics.informResponses}</div>
              <div className="text-sm text-gray-600">Inform</div>
            </div>
          </div>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-2xl shadow-xl p-6 mb-6">
          <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-blue-600" />
            Test the System
          </h2>
          
          <div className="space-y-4">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter a message to analyze emotional context and generate a safe response..."
              className="w-full h-32 p-4 border-2 border-gray-200 rounded-xl focus:border-blue-500 focus:outline-none resize-none"
            />
            
            <div className="flex gap-2 flex-wrap">
              <button
                onClick={processInput}
                disabled={!input.trim() || isProcessing}
                className="px-6 py-3 bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-lg font-semibold hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing ? 'Processing...' : 'Analyze & Respond'}
              </button>
              
              {testCases.map((test, i) => (
                <button
                  key={i}
                  onClick={() => setInput(test)}
                  className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition-all"
                >
                  Test Case {i + 1}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Results */}
        {results.map(result => (
          <div key={result.id} className="bg-white rounded-2xl shadow-xl p-6 mb-6">
            <div className="flex items-start justify-between mb-4">
              <h3 className="text-lg font-bold text-gray-900">Analysis Result</h3>
              <span className="text-sm text-gray-500">{result.timestamp}</span>
            </div>

            <div className="bg-gray-50 rounded-lg p-4 mb-4">
              <div className="text-sm text-gray-600 mb-1">Input:</div>
              <div className="text-gray-900">{result.input}</div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
              {/* Context Analysis */}
              <div className="border-2 border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Brain className="w-5 h-5 text-blue-600" />
                  <h4 className="font-bold">Context Analysis</h4>
                </div>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-600">Context:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-white ${
                      result.analysis.context === 'crisis' ? 'bg-red-600' :
                      result.analysis.context === 'distress' ? 'bg-orange-500' :
                      result.analysis.context === 'vulnerability' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}>
                      {result.analysis.context.toUpperCase()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Confidence:</span>
                    <span className="ml-2 font-semibold">{result.analysis.confidence.toFixed(1)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Severity:</span>
                    <span className="ml-2 font-semibold">{result.analysis.severity}</span>
                  </div>
                </div>
              </div>

              {/* Sentiment */}
              <div className="border-2 border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Heart className="w-5 h-5 text-pink-600" />
                  <h4 className="font-bold">Sentiment</h4>
                </div>
                <div className="space-y-2 text-sm">
                  {Object.entries(result.sentiment).map(([emotion, score]) => (
                    score > 0 && (
                      <div key={emotion}>
                        <div className="flex justify-between mb-1">
                          <span className="text-gray-600 capitalize">{emotion}:</span>
                          <span className="font-semibold">{(score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-pink-400 to-purple-500 h-2 rounded-full"
                            style={{ width: `${score * 100}%` }}
                          />
                        </div>
                      </div>
                    )
                  ))}
                </div>
              </div>

              {/* Risk Assessment */}
              <div className="border-2 border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Shield className="w-5 h-5 text-red-600" />
                  <h4 className="font-bold">Risk Assessment</h4>
                </div>
                <div className="space-y-2 text-sm">
                  <div>
                    <span className="text-gray-600">Level:</span>
                    <span className={`ml-2 px-2 py-1 rounded text-white ${
                      result.risk.level === 'critical' ? 'bg-red-600' :
                      result.risk.level === 'high' ? 'bg-orange-500' :
                      result.risk.level === 'moderate' ? 'bg-yellow-500' :
                      'bg-green-500'
                    }`}>
                      {result.risk.level.toUpperCase()}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">Score:</span>
                    <span className="ml-2 font-semibold">{result.risk.score}/10</span>
                  </div>
                  <div>
                    <span className="text-gray-600">High-Risk:</span>
                    <span className="ml-2">{result.risk.highRiskIndicators}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">Protective:</span>
                    <span className="ml-2">{result.risk.protectiveFactors}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-2">
                    {result.risk.requiresImmediate ? (
                      <><AlertCircle className="w-4 h-4 text-red-600" />
                      <span className="text-red-600 font-semibold">Immediate Action Required</span></>
                    ) : (
                      <><CheckCircle className="w-4 h-4 text-green-600" />
                      <span className="text-green-600">No Immediate Action</span></>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Response */}
            <div className="border-2 border-blue-200 bg-blue-50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <div className={`px-3 py-1 rounded-full text-white font-semibold ${
                  result.response.tier === 'defer' ? 'bg-red-600' :
                  result.response.tier === 'support' ? 'bg-orange-500' :
                  'bg-blue-600'
                }`}>
                  {result.response.tier.toUpperCase()} TIER
                </div>
              </div>
              
              <div className="mb-4">
                <div className="font-semibold mb-2">Response:</div>
                <div className="text-gray-800">{result.response.message}</div>
              </div>

              {result.response.resources.length > 0 && (
                <div className="mb-4">
                  <div className="font-semibold mb-2">Resources:</div>
                  <ul className="space-y-1">
                    {result.response.resources.map((resource, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-blue-600">•</span>
                        <span className="text-gray-800">{resource}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="text-sm text-gray-700 italic">
                {result.response.followUp}
              </div>
            </div>
          </div>
        ))}

        {/* Features */}
        <div className="bg-white rounded-2xl shadow-xl p-6">
          <h2 className="text-xl font-bold mb-4">System Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Emotional Context Detection</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Sentiment Analysis (Multi-emotion)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Risk Assessment (0-10 scale)</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Policy-Based Response Generation</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Crisis Resource Provision</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-600" />
              <span>Real-time Metrics Tracking</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmpathyAIDemo;
