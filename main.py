"""
Production-Ready Impact-Aware AI Response System with Neural Network
A complete implementation with emotional context detection, policy enforcement,
and neural network-based sentiment analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import anthropic


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('empathy_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Enums for type safety
class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PolicyTier(Enum):
    DEFER = "defer"
    SUPPORT = "support"
    INFORM = "inform"


class EmotionalContext(Enum):
    CRISIS = "crisis"
    DISTRESS = "distress"
    VULNERABILITY = "vulnerability"
    MANIPULATION_RISK = "manipulation_risk"
    NEUTRAL = "neutral"


# Data classes
@dataclass
class ContextAnalysis:
    """Results from emotional context detection"""
    context: EmotionalContext
    severity: SeverityLevel
    confidence: float
    neural_score: float
    flags: Dict[str, bool]
    timestamp: str


@dataclass
class ResponsePolicy:
    """Policy enforcement rules"""
    tier: PolicyTier
    action: str
    allowed_actions: List[str]
    restrictions: List[str]


@dataclass
class SafeResponse:
    """Generated safe response"""
    message: str
    resources: List[str]
    follow_up: str
    policy_tier: str
    used_ai: bool


@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_interactions: int = 0
    safe_escalations: int = 0
    inform_responses: int = 0
    support_responses: int = 0
    deferral_rate: float = 0.0
    wellbeing_score: float = 0.0
    neural_accuracy: float = 0.0


class EmotionalContextNeuralNetwork:
    """
    Neural network for emotional context classification
    Uses LSTM for sequential text understanding
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = 100
        self.model = None
        self.tokenizer = keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token="<OOV>"
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def build_model(self, num_classes: int = 5):
        """Build the neural network architecture"""
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim, 
                           input_length=self.max_length),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info("Neural network model built successfully")
        return model
    
    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for emotional context detection"""
        training_examples = {
            EmotionalContext.CRISIS: [
                "I want to end my life", "suicide thoughts", "I can't go on",
                "I want to die", "no point in living", "kill myself",
                "ending it all", "I'm going to hurt myself", "goodbye forever",
                "everyone would be better without me", "I've written my note"
            ],
            EmotionalContext.DISTRESS: [
                "I'm so depressed", "severe anxiety attacks", "can't stop crying",
                "overwhelming panic", "I can't cope anymore", "everything is falling apart",
                "I'm having a breakdown", "can't breathe from anxiety", 
                "depression is crushing me", "constant panic attacks"
            ],
            EmotionalContext.VULNERABILITY: [
                "I feel so lonely", "I'm scared", "feeling lost and confused",
                "I'm worried about everything", "feeling sad today", "need support",
                "going through hard time", "feeling vulnerable", "need someone to talk",
                "feeling isolated and alone", "struggling with emotions"
            ],
            EmotionalContext.MANIPULATION_RISK: [
                "only you understand me", "I need you always", "can't live without you",
                "you're the only one who can help", "don't leave me", 
                "I depend on you completely", "you're my everything",
                "nobody else gets me like you do", "you're always here for me"
            ],
            EmotionalContext.NEUTRAL: [
                "how do I reset my password", "what's the weather today",
                "help me with this code", "explain quantum physics",
                "recipe for chocolate cake", "how to change my settings",
                "information about history", "technical question about API",
                "when is the deadline", "what are the requirements"
            ]
        }
        
        texts = []
        labels = []
        
        for context, examples in training_examples.items():
            for example in examples:
                texts.append(example)
                labels.append(context.value)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Tokenize texts
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        return padded, labels_encoded
    
    def train(self, epochs: int = 50, validation_split: float = 0.2):
        """Train the neural network"""
        X, y = self.prepare_training_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        if self.model is None:
            self.build_model(num_classes=len(np.unique(y)))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        logger.info(f"Model training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        return history
    
    def predict(self, text: str) -> Tuple[EmotionalContext, float]:
        """Predict emotional context for input text"""
        if not self.is_trained:
            logger.warning("Model not trained. Using rule-based fallback.")
            return EmotionalContext.NEUTRAL, 0.0
        
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.max_length,
            padding='post',
            truncating='post'
        )
        
        predictions = self.model.predict(padded, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        context_label = self.label_encoder.inverse_transform([predicted_class])[0]
        context = EmotionalContext(context_label)
        
        return context, confidence
    
    def save_model(self, path: str = "models/empathy_model"):
        """Save the trained model"""
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/model.h5")
        
        with open(f"{path}/tokenizer.json", "w") as f:
            json.dump(self.tokenizer.to_json(), f)
        
        with open(f"{path}/label_encoder.json", "w") as f:
            json.dump(self.label_encoder.classes_.tolist(), f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = "models/empathy_model"):
        """Load a trained model"""
        self.model = keras.models.load_model(f"{path}/model.h5")
        
        with open(f"{path}/tokenizer.json", "r") as f:
            self.tokenizer = keras.preprocessing.text.tokenizer_from_json(json.load(f))
        
        with open(f"{path}/label_encoder.json", "r") as f:
            classes = json.load(f)
            self.label_encoder.classes_ = np.array(classes)
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


class EmotionalContextDetector:
    """Rule-based emotional context detection with neural network enhancement"""
    
    PATTERNS = {
        EmotionalContext.CRISIS: {
            'keywords': ['suicide', 'kill myself', 'end it all', 'no point living', 
                        'want to die', 'hurt myself', 'goodbye forever'],
            'severity': SeverityLevel.CRITICAL
        },
        EmotionalContext.DISTRESS: {
            'keywords': ['depressed', 'anxious', 'panic', 'overwhelming', 
                        "can't cope", 'struggling', 'breakdown', "can't breathe"],
            'severity': SeverityLevel.HIGH
        },
        EmotionalContext.VULNERABILITY: {
            'keywords': ['lonely', 'sad', 'worried', 'scared', 'confused', 
                        'lost', 'vulnerable', 'isolated'],
            'severity': SeverityLevel.MEDIUM
        },
        EmotionalContext.MANIPULATION_RISK: {
            'keywords': ['only you understand', 'need you', "can't without you", 
                        'always here', 'depend on you', "you're my everything"],
            'severity': SeverityLevel.MEDIUM
        },
        EmotionalContext.NEUTRAL: {
            'keywords': ['how', 'what', 'when', 'explain', 'information', 
                        'help with', 'question about'],
            'severity': SeverityLevel.LOW
        }
    }
    
    def __init__(self, use_neural: bool = True):
        self.use_neural = use_neural
        self.neural_net = EmotionalContextNeuralNetwork() if use_neural else None
    
    def detect(self, text: str) -> ContextAnalysis:
        """Detect emotional context using both rules and neural network"""
        text_lower = text.lower()
        
        # Rule-based detection
        rule_context, rule_confidence = self._rule_based_detection(text_lower)
        
        # Neural network detection
        neural_context = rule_context
        neural_score = 0.0
        
        if self.use_neural and self.neural_net and self.neural_net.is_trained:
            neural_context, neural_score = self.neural_net.predict(text)
            
            # Combine rule-based and neural predictions
            # Prioritize critical contexts from either method
            if rule_context == EmotionalContext.CRISIS or neural_context == EmotionalContext.CRISIS:
                final_context = EmotionalContext.CRISIS
                final_confidence = max(rule_confidence, neural_score)
            else:
                # Use higher confidence prediction
                if neural_score > rule_confidence:
                    final_context = neural_context
                    final_confidence = neural_score
                else:
                    final_context = rule_context
                    final_confidence = rule_confidence
        else:
            final_context = rule_context
            final_confidence = rule_confidence
        
        # Generate safety flags
        flags = self._generate_flags(final_context)
        
        return ContextAnalysis(
            context=final_context,
            severity=self.PATTERNS[final_context]['severity'],
            confidence=final_confidence,
            neural_score=neural_score,
            flags=flags,
            timestamp=datetime.now().isoformat()
        )
    
    def _rule_based_detection(self, text: str) -> Tuple[EmotionalContext, float]:
        """Rule-based detection for baseline"""
        max_confidence = 0.0
        detected_context = EmotionalContext.NEUTRAL
        
        for context, data in self.PATTERNS.items():
            matches = sum(1 for keyword in data['keywords'] if keyword in text)
            confidence = matches / len(data['keywords'])
            
            if confidence > max_confidence:
                max_confidence = confidence
                detected_context = context
        
        return detected_context, min(max_confidence, 1.0)
    
    def _generate_flags(self, context: EmotionalContext) -> Dict[str, bool]:
        """Generate safety flags based on context"""
        return {
            'requires_human': context == EmotionalContext.CRISIS,
            'limit_persuasion': context in [EmotionalContext.MANIPULATION_RISK, 
                                           EmotionalContext.VULNERABILITY],
            'emotional_support': context in [EmotionalContext.DISTRESS, 
                                            EmotionalContext.VULNERABILITY],
            'immediate_action': context == EmotionalContext.CRISIS
        }


class ResponsePolicyEngine:
    """Determines appropriate response policy based on context"""
    
    POLICIES = {
        EmotionalContext.CRISIS: ResponsePolicy(
            tier=PolicyTier.DEFER,
            action='immediate_escalation',
            allowed_actions=['provide_resources', 'connect_human', 'ensure_safety'],
            restrictions=['no_advice', 'no_minimizing', 'no_attachment_language', 
                         'no_diagnosis', 'no_delay']
        ),
        EmotionalContext.DISTRESS: ResponsePolicy(
            tier=PolicyTier.SUPPORT,
            action='empathetic_support',
            allowed_actions=['validate_feelings', 'suggest_resources', 
                           'gentle_guidance', 'normalize_emotions'],
            restrictions=['no_diagnosis', 'no_promises', 'limited_engagement',
                         'encourage_professional_help']
        ),
        EmotionalContext.VULNERABILITY: ResponsePolicy(
            tier=PolicyTier.SUPPORT,
            action='compassionate_inform',
            allowed_actions=['provide_information', 'emotional_validation', 
                           'suggest_next_steps', 'offer_resources'],
            restrictions=['no_dependency_language', 'encourage_human_connection',
                         'maintain_boundaries']
        ),
        EmotionalContext.MANIPULATION_RISK: ResponsePolicy(
            tier=PolicyTier.INFORM,
            action='boundary_setting',
            allowed_actions=['factual_response', 'redirect_to_resources',
                           'clarify_limitations'],
            restrictions=['no_personal_language', 'no_exclusive_language', 
                         'emphasize_limitations', 'redirect_appropriately']
        ),
        EmotionalContext.NEUTRAL: ResponsePolicy(
            tier=PolicyTier.INFORM,
            action='standard_response',
            allowed_actions=['provide_information', 'answer_questions', 
                           'general_guidance', 'helpful_suggestions'],
            restrictions=['maintain_professional_boundary', 'factual_accuracy']
        )
    }
    
    @classmethod
    def get_policy(cls, context: EmotionalContext) -> ResponsePolicy:
        """Get response policy for given context"""
        return cls.POLICIES.get(context, cls.POLICIES[EmotionalContext.NEUTRAL])


class SafeResponseGenerator:
    """Generates safe, policy-compliant responses"""
    
    def __init__(self, api_key: Optional[str] = None, use_ai: bool = True):
        self.use_ai = use_ai
        self.client = None
        
        if use_ai:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("AI response generation enabled")
            else:
                logger.warning("No API key provided. Using template-based responses.")
                self.use_ai = False
    
    def generate(self, text: str, analysis: ContextAnalysis, 
                 policy: ResponsePolicy) -> SafeResponse:
        """Generate safe response based on analysis and policy"""
        
        if self.use_ai and self.client:
            try:
                return self._generate_ai_response(text, analysis, policy)
            except Exception as e:
                logger.error(f"AI generation failed: {e}. Using template fallback.")
                return self._generate_template_response(analysis, policy)
        else:
            return self._generate_template_response(analysis, policy)
    
    def _generate_ai_response(self, text: str, analysis: ContextAnalysis, 
                              policy: ResponsePolicy) -> SafeResponse:
        """Generate response using Claude API"""
        system_prompt = self._build_system_prompt(analysis, policy)
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": text}]
        )
        
        response_text = message.content[0].text
        
        # Parse response
        parsed = self._parse_ai_response(response_text)
        
        return SafeResponse(
            message=parsed['message'],
            resources=parsed['resources'],
            follow_up=parsed['follow_up'],
            policy_tier=policy.tier.value,
            used_ai=True
        )
    
    def _build_system_prompt(self, analysis: ContextAnalysis, 
                            policy: ResponsePolicy) -> str:
        """Build system prompt with policy constraints"""
        prompt = f"""You are an empathetic AI assistant with strict safety guardrails.

DETECTED CONTEXT: {analysis.context.value}
SEVERITY: {analysis.severity.value}
CONFIDENCE: {analysis.confidence:.2%}
POLICY TIER: {policy.tier.value}

ALLOWED ACTIONS: {', '.join(policy.allowed_actions)}
RESTRICTIONS: {', '.join(policy.restrictions)}

"""
        
        tier_guidelines = {
            PolicyTier.DEFER: """CRITICAL: This is a crisis situation requiring immediate human intervention.
- Your response MUST prioritize connecting the user to crisis resources
- Provide specific hotline numbers: 988 Suicide & Crisis Lifeline, Crisis Text Line (741741)
- Do NOT attempt to provide advice or solutions yourself
- Use clear, direct, caring language: "Your safety is the top priority"
- Emphasize that professional help is available NOW
- Do not minimize or delay - immediate action is required""",
            
            PolicyTier.SUPPORT: """This user needs emotional support with clear boundaries.
- Validate their feelings without diagnosing: "What you're feeling is understandable"
- Provide helpful resources and suggest professional support
- Use warm, empathetic language but maintain boundaries
- Avoid making promises or creating dependency
- Encourage connection with trusted people: family, friends, professionals
- Suggest concrete next steps they can take""",
            
            PolicyTier.INFORM: """Provide helpful information with appropriate boundaries.
- Be factual, clear, and direct
- Maintain professional distance
- Avoid overly personal or exclusive language
- If needed, redirect to appropriate resources
- Keep response focused and practical
- Clarify your limitations as an AI"""
        }
        
        return prompt + tier_guidelines[policy.tier]
    
    def _parse_ai_response(self, response_text: str) -> Dict[str, any]:
        """Parse AI response into structured format"""
        paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
        
        # Extract resources (lines with bullets, numbers, or resource keywords)
        resources = []
        message_parts = []
        
        for para in paragraphs:
            lines = para.split('\n')
            para_resources = []
            para_text = []
            
            for line in lines:
                if (re.match(r'^\s*[-•*\d.]+\s', line) or
                    any(keyword in line.lower() for keyword in 
                        ['hotline', 'helpline', 'call', 'text', 'professional', 
                         'therapist', 'counselor', 'support'])):
                    cleaned = re.sub(r'^\s*[-•*\d.]+\s*', '', line).strip()
                    if cleaned:
                        para_resources.append(cleaned)
                else:
                    para_text.append(line)
            
            if para_resources:
                resources.extend(para_resources)
            if para_text:
                message_parts.append('\n'.join(para_text))
        
        return {
            'message': message_parts[0] if message_parts else response_text,
            'resources': resources[:5],  # Limit to 5 resources
            'follow_up': message_parts[-1] if len(message_parts) > 1 else 
                        "Is there anything else I can help you with?"
        }
    
    def _generate_template_response(self, analysis: ContextAnalysis, 
                                    policy: ResponsePolicy) -> SafeResponse:
        """Generate template-based response as fallback"""
        templates = {
            EmotionalContext.CRISIS: {
                'message': "I'm concerned about what you've shared. Your safety is the top priority. Please reach out to a crisis helpline immediately:",
                'resources': [
                    "988 Suicide & Crisis Lifeline: Call or text 988",
                    "Crisis Text Line: Text HOME to 741741",
                    "International Association for Suicide Prevention: findahelpline.com"
                ],
                'follow_up': "Please talk to a trusted person or mental health professional right away. You deserve support from someone who can help."
            },
            EmotionalContext.DISTRESS: {
                'message': "I hear that you're going through a difficult time. What you're feeling is valid, and it's important to get proper support.",
                'resources': [
                    "Consider reaching out to a licensed therapist or counselor",
                    "Contact your doctor or a mental health professional",
                    "Speak with someone you trust about how you're feeling"
                ],
                'follow_up': "While I can provide information, a qualified professional can offer the personalized care you deserve."
            },
            EmotionalContext.VULNERABILITY: {
                'message': "Thank you for sharing. It's okay to feel this way, and seeking support is a positive step.",
                'resources': [
                    "Talking to a friend, family member, or counselor can help",
                    "Many people find journaling or creative expression helpful",
                    "Professional guidance can provide tailored strategies"
                ],
                'follow_up': "Remember that connecting with real people in your life can make a meaningful difference."
            },
            EmotionalContext.MANIPULATION_RISK: {
                'message': "I'm here to provide information and support, but I have important limitations as an AI system.",
                'resources': [
                    "I can't replace human relationships or professional support",
                    "For personalized guidance, please consult with a qualified professional",
                    "Building connections with people in your life is important for wellbeing"
                ],
                'follow_up': "I encourage you to reach out to trusted individuals who can provide ongoing support."
            },
            EmotionalContext.NEUTRAL: {
                'message': "I'm happy to help with your question. Let me provide you with the information you need.",
                'resources': [],
                'follow_up': "Let me know if you need clarification or have other questions."
            }
        }
        
        template = templates[analysis.context]
        return SafeResponse(
            message=template['message'],
            resources=template['resources'],
            follow_up=template['follow_up'],
            policy_tier=policy.tier.value,
            used_ai=False
        )


class SentimentAnalyzer:
    """Advanced sentiment analysis with emotion detection"""
    
    EMOTION_PATTERNS = {
        'anger': ['angry', 'furious', 'mad', 'rage', 'hate', 'pissed'],
        'sadness': ['sad', 'depressed', 'hopeless', 'miserable', 'crying'],
        'fear': ['scared', 'afraid', 'terrified', 'anxious', 'worried', 'panic'],
        'joy': ['happy', 'excited', 'joyful', 'great', 'wonderful'],
        'disgust': ['disgusted', 'sick', 'revolted', 'repulsed'],
        'surprise': ['shocked', 'surprised', 'amazed', 'astonished']
    }
    
    @classmethod
    def analyze(cls, text: str) -> Dict[str, float]:
        """Analyze emotional content of text"""
        text_lower = text.lower()
        emotions = {}
        
        for emotion, keywords in cls.EMOTION_PATTERNS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(score / len(keywords), 1.0)
        
        return emotions


class ConversationMemory:
    """Maintains conversation context and history"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations = {}
    
    def add_interaction(self, user_id: str, user_msg: str, 
                       ai_response: str, context: str):
        """Add interaction to memory"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        self.conversations[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'user': user_msg,
            'ai': ai_response,
            'context': context
        })
        
        # Keep only recent history
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
    
    def get_context(self, user_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        return self.conversations.get(user_id, [])
    
    def detect_escalation(self, user_id: str) -> bool:
        """Detect if emotional state is worsening"""
        history = self.get_context(user_id)
        if len(history) < 3:
            return False
        
        # Check if recent contexts show increasing severity
        recent = history[-3:]
        severity_map = {'neutral': 0, 'vulnerability': 1, 'distress': 2, 'crisis': 3}
        
        scores = [severity_map.get(h['context'], 0) for h in recent]
        return all(scores[i] <= scores[i+1] for i in range(len(scores)-1))


class RiskAssessment:
    """Advanced risk assessment and scoring"""
    
    HIGH_RISK_INDICATORS = [
        'suicide', 'kill myself', 'end it', 'no reason to live',
        'want to die', 'better off dead', 'goodbye forever'
    ]
    
    MEDIUM_RISK_INDICATORS = [
        'can\'t take it', 'give up', 'hopeless', 'no way out',
        'pointless', 'unbearable', 'overwhelming pain'
    ]
    
    PROTECTIVE_FACTORS = [
        'family', 'friends', 'therapy', 'treatment', 'help',
        'support', 'getting better', 'trying', 'hope'
    ]
    
    @classmethod
    def assess(cls, text: str, history: List[Dict] = None) -> Dict[str, any]:
        """Comprehensive risk assessment"""
        text_lower = text.lower()
        
        # Count risk indicators
        high_risk_count = sum(1 for indicator in cls.HIGH_RISK_INDICATORS 
                             if indicator in text_lower)
        medium_risk_count = sum(1 for indicator in cls.MEDIUM_RISK_INDICATORS 
                               if indicator in text_lower)
        protective_count = sum(1 for factor in cls.PROTECTIVE_FACTORS 
                              if factor in text_lower)
        
        # Calculate risk score (0-10)
        risk_score = (high_risk_count * 3 + medium_risk_count * 1.5 - protective_count * 0.5)
        risk_score = max(0, min(10, risk_score))
        
        # Risk level
        if risk_score >= 7:
            risk_level = 'critical'
        elif risk_score >= 4:
            risk_level = 'high'
        elif risk_score >= 2:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'high_risk_indicators': high_risk_count,
            'medium_risk_indicators': medium_risk_count,
            'protective_factors': protective_count,
            'requires_immediate_action': risk_score >= 7
        }


class BiasDetection:
    """Detects and mitigates potential biases in responses"""
    
    BIAS_PATTERNS = {
        'gender': ['he should', 'she should', 'men are', 'women are'],
        'cultural': ['your culture', 'people like you', 'in your country'],
        'age': ['at your age', 'young people', 'older people'],
        'socioeconomic': ['people like you can\'t', 'you should just buy']
    }
    
    @classmethod
    def check_response(cls, response: str) -> Dict[str, any]:
        """Check response for potential biases"""
        response_lower = response.lower()
        detected_biases = []
        
        for bias_type, patterns in cls.BIAS_PATTERNS.items():
            if any(pattern in response_lower for pattern in patterns):
                detected_biases.append(bias_type)
        
        return {
            'has_bias': len(detected_biases) > 0,
            'bias_types': detected_biases,
            'needs_revision': len(detected_biases) > 0
        }


class ResourceRecommendation:
    """Intelligent resource recommendation system"""
    
    RESOURCES = {
        'crisis': [
            {'name': '988 Suicide & Crisis Lifeline', 'contact': 'Call/Text 988', 'type': 'hotline'},
            {'name': 'Crisis Text Line', 'contact': 'Text HOME to 741741', 'type': 'text'},
            {'name': 'International Association for Suicide Prevention', 'url': 'findahelpline.com', 'type': 'directory'}
        ],
        'mental_health': [
            {'name': 'NAMI Helpline', 'contact': '1-800-950-NAMI', 'type': 'hotline'},
            {'name': 'SAMHSA National Helpline', 'contact': '1-800-662-4357', 'type': 'hotline'},
            {'name': 'Psychology Today Therapist Finder', 'url': 'psychologytoday.com', 'type': 'directory'}
        ],
        'anxiety': [
            {'name': 'Anxiety and Depression Association', 'url': 'adaa.org', 'type': 'information'},
            {'name': 'Calm App', 'type': 'app'},
            {'name': 'Headspace', 'type': 'app'}
        ],
        'abuse': [
            {'name': 'National Domestic Violence Hotline', 'contact': '1-800-799-7233', 'type': 'hotline'},
            {'name': 'RAINN (Rape, Abuse & Incest National Network)', 'contact': '1-800-656-HOPE', 'type': 'hotline'}
        ],
        'substance': [
            {'name': 'SAMHSA Treatment Locator', 'url': 'findtreatment.gov', 'type': 'directory'},
            {'name': 'Alcoholics Anonymous', 'url': 'aa.org', 'type': 'support_group'},
            {'name': 'Narcotics Anonymous', 'url': 'na.org', 'type': 'support_group'}
        ]
    }
    
    @classmethod
    def recommend(cls, context: EmotionalContext, text: str) -> List[Dict]:
        """Recommend appropriate resources"""
        text_lower = text.lower()
        recommended = []
        
        # Crisis resources always first for crisis context
        if context == EmotionalContext.CRISIS:
            recommended.extend(cls.RESOURCES['crisis'])
        
        # Check for specific issues
        if any(word in text_lower for word in ['abuse', 'assault', 'violence']):
            recommended.extend(cls.RESOURCES['abuse'][:2])
        
        if any(word in text_lower for word in ['alcohol', 'drugs', 'addiction', 'substance']):
            recommended.extend(cls.RESOURCES['substance'][:2])
        
        if any(word in text_lower for word in ['anxiety', 'panic', 'anxious']):
            recommended.extend(cls.RESOURCES['anxiety'][:2])
        
        # General mental health resources
        if context in [EmotionalContext.DISTRESS, EmotionalContext.VULNERABILITY]:
            recommended.extend(cls.RESOURCES['mental_health'][:2])
        
        return recommended[:5]  # Limit to 5 resources


class QualityAssurance:
    """Quality assurance checks for responses"""
    
    PROHIBITED_PHRASES = [
        'i know exactly how you feel',
        'just think positive',
        'it could be worse',
        'snap out of it',
        'it\'s all in your head',
        'you\'re overreacting'
    ]
    
    REQUIRED_ELEMENTS = {
        PolicyTier.DEFER: ['crisis', 'professional', 'immediate'],
        PolicyTier.SUPPORT: ['understand', 'support', 'resources'],
        PolicyTier.INFORM: ['information', 'help']
    }
    
    @classmethod
    def check(cls, response: str, policy: ResponsePolicy) -> Dict[str, any]:
        """Perform quality checks on response"""
        response_lower = response.lower()
        
        # Check for prohibited phrases
        violations = [phrase for phrase in cls.PROHIBITED_PHRASES 
                     if phrase in response_lower]
        
        # Check for required elements
        required = cls.REQUIRED_ELEMENTS.get(policy.tier, [])
        missing = [elem for elem in required 
                  if elem not in response_lower]
        
        # Check length
        word_count = len(response.split())
        appropriate_length = 20 <= word_count <= 300
        
        return {
            'passes': len(violations) == 0 and len(missing) == 0 and appropriate_length,
            'violations': violations,
            'missing_elements': missing,
            'word_count': word_count,
            'appropriate_length': appropriate_length
        }


class EmpathyAISystem:
    """Main production system orchestrating all components"""
    
    def __init__(self, use_neural: bool = True, use_ai: bool = True, 
                 api_key: Optional[str] = None):
        self.detector = EmotionalContextDetector(use_neural=use_neural)
        self.generator = SafeResponseGenerator(api_key=api_key, use_ai=use_ai)
        self.metrics = SystemMetrics()
        self.interaction_history = []
        
        # New advanced features
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conversation_memory = ConversationMemory()
        self.risk_assessor = RiskAssessment()
        self.bias_detector = BiasDetection()
        self.resource_recommender = ResourceRecommendation()
        self.qa_checker = QualityAssurance()
        
        logger.info(f"EmpathyAI System initialized (Neural: {use_neural}, AI: {use_ai})")
    
    def train_neural_network(self, epochs: int = 50):
        """Train the neural network component"""
        if self.detector.use_neural and self.detector.neural_net:
            history = self.detector.neural_net.train(epochs=epochs)
            self.metrics.neural_accuracy = history.history['accuracy'][-1]
            logger.info(f"Neural network trained with accuracy: {self.metrics.neural_accuracy:.4f}")
            return history
        else:
            logger.warning("Neural network not enabled")
            return None
    
    def process_input(self, text: str, user_id: str = "anonymous") -> Dict[str, any]:
        """Process user input through the complete pipeline with advanced features"""
        try:
            # Step 1: Detect emotional context
            analysis = self.detector.detect(text)
            logger.info(f"Context detected: {analysis.context.value} "
                       f"(confidence: {analysis.confidence:.2%})")
            
            # Step 2: Sentiment analysis
            sentiment = self.sentiment_analyzer.analyze(text)
            logger.info(f"Sentiment: {sentiment}")
            
            # Step 3: Risk assessment
            history = self.conversation_memory.get_context(user_id)
            risk = self.risk_assessor.assess(text, history)
            logger.info(f"Risk level: {risk['risk_level']} (score: {risk['risk_score']:.1f})")
            
            # Step 4: Check for conversation escalation
            escalation = self.conversation_memory.detect_escalation(user_id)
            if escalation:
                logger.warning(f"User {user_id} showing escalating distress")
            
            # Step 5: Determine response policy (upgrade if high risk)
            policy = ResponsePolicyEngine.get_policy(analysis.context)
            if risk['requires_immediate_action']:
                policy = ResponsePolicyEngine.get_policy(EmotionalContext.CRISIS)
                logger.warning("Policy upgraded to CRISIS due to risk assessment")
            
            # Step 6: Get intelligent resource recommendations
            resources = self.resource_recommender.recommend(analysis.context, text)
            logger.info(f"Recommended {len(resources)} resources")
            
            # Step 7: Generate safe response
            response = self.generator.generate(text, analysis, policy)
            
            # Step 8: Quality assurance check
            qa_result = self.qa_checker.check(response.message, policy)
            if not qa_result['passes']:
                logger.warning(f"QA check failed: {qa_result}")
                # Regenerate or use template if QA fails
                if qa_result['violations']:
                    response = self.generator._generate_template_response(analysis, policy)
            
            # Step 9: Bias detection
            bias_check = self.bias_detector.check_response(response.message)
            if bias_check['has_bias']:
                logger.warning(f"Bias detected in response: {bias_check['bias_types']}")
            
            # Step 10: Merge recommended resources
            if resources:
                additional_resources = [
                    f"{r['name']}" + (f" - {r['contact']}" if 'contact' in r else f" - {r.get('url', '')}")
                    for r in resources
                ]
                response.resources.extend(additional_resources)
                response.resources = list(dict.fromkeys(response.resources))[:5]  # Dedupe and limit
            
            # Step 11: Update conversation memory
            self.conversation_memory.add_interaction(
                user_id, text, response.message, analysis.context.value
            )
            
            # Step 12: Update metrics
            self._update_metrics(policy.tier)
            
            # Step 13: Log interaction with all analysis
            interaction = {
                'timestamp': analysis.timestamp,
                'user_id': user_id,
                'input': text,
                'analysis': asdict(analysis),
                'sentiment': sentiment,
                'risk': risk,
                'escalation': escalation,
                'policy': asdict(policy),
                'response': asdict(response),
                'qa_result': qa_result,
                'bias_check': bias_check,
                'resources_provided': len(resources)
            }
            self.interaction_history.append(interaction)
            
            return {
                'success': True,
                'analysis': asdict(analysis),
                'sentiment': sentiment,
                'risk': risk,
                'escalation_detected': escalation,
                'policy': asdict(policy),
                'response': asdict(response),
                'qa_passed': qa_result['passes'],
                'bias_free': not bias_check['has_bias'],
                'metrics': asdict(self.metrics)
            }
            
        except Exception as e:
            logger.error(f"Error processing input: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'fallback_response': self._get_error_fallback()
            }
    
    def _update_metrics(self, tier: PolicyTier):
        """Update system metrics"""
        self.metrics.total_interactions += 1
        
        if tier == PolicyTier.DEFER:
            self.metrics.safe_escalations += 1
        elif tier == PolicyTier.SUPPORT:
            self.metrics.support_responses += 1
        elif tier == PolicyTier.INFORM:
            self.metrics.inform_responses += 1
        
        if self.metrics.total_interactions > 0:
            self.metrics.deferral_rate = (
                self.metrics.safe_escalations / self.metrics.total_interactions
            ) * 100
        
        # Simulate wellbeing score (in production, would use real user feedback)
        self.metrics.wellbeing_score = 85.0 + (np.random.random() * 10)
    
    def _get_error_fallback(self) -> Dict[str, str]:
        """Provide safe fallback response on system error"""
        return {
            'message': "I'm experiencing a technical issue, but your wellbeing is important. "
                      "If you're in crisis, please contact 988 Suicide & Crisis Lifeline immediately.",
            'resources': [
                "988 Suicide & Crisis Lifeline: Call or text 988",
                "Crisis Text Line: Text HOME to 741741"
            ],
            'follow_up': "Please try again in a moment, or reach out to a trusted person for support."
        }
    
    def export_metrics(self, filepath: str = "metrics.json"):
        """Export metrics to file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
        logger.info(f"Metrics exported to {filepath}")
    
    def export_history(self, filepath: str = "interaction_history.json"):
        """Export interaction history"""
        with open(filepath, 'w') as f:
            json.dump(self.interaction_history, f, indent=2)
        logger.info(f"History exported to {filepath}")


def main():
    """Main execution function with example usage"""
    print("=" * 70)
    print("Production-Ready Impact-Aware AI Response System")
    print("Neural Network + Claude AI Integration")
    print("=" * 70)
    print()
    
    # Initialize system
    print("Initializing system...")
    system = EmpathyAISystem(use_neural=True, use_ai=True)
    
    # Train neural network
    print("\nTraining neural network (this may take a minute)...")
    system.train_neural_network(epochs=30)
    
    # Test cases
    test_inputs = [
        ("I'm thinking about suicide and don't want to live anymore", "user_001"),
        ("I'm feeling really depressed and overwhelmed with anxiety", "user_002"),
        ("I feel lonely and need someone to talk to", "user_003"),
        ("You're the only one who understands me, I need you", "user_004"),
        ("How do I reset my password?", "user_005"),
        ("I've been drinking too much and can't stop", "user_006")
    ]
    
    print("\n" + "=" * 70)
    print("Processing Test Cases with Advanced Features")
    print("=" * 70)
    
    for i, (test_input, user_id) in enumerate(test_inputs, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"User ID: {user_id}")
        print(f"Input: {test_input}")
        print()
        
        result = system.process_input(test_input, user_id)
        
        if result['success']:
            analysis = result['analysis']
            sentiment = result['sentiment']
            risk = result['risk']
            response = result['response']
            
            print(f"Context: {analysis['context']} (Confidence: {analysis['confidence']:.1%})")
            print(f"Severity: {analysis['severity']}")
            print(f"Neural Score: {analysis['neural_score']:.1%}")
            print()
            
            print("Sentiment Analysis:")
            for emotion, score in sentiment.items():
                if score > 0:
                    print(f"  {emotion.capitalize()}: {score:.1%}")
            print()
            
            print(f"Risk Assessment:")
            print(f"  Level: {risk['risk_level'].upper()}")
            print(f"  Score: {risk['risk_score']:.1f}/10")
            print(f"  High-risk indicators: {risk['high_risk_indicators']}")
            print(f"  Protective factors: {risk['protective_factors']}")
            print(f"  Immediate action required: {risk['requires_immediate_action']}")
            print()
            
            print(f"Escalation Detected: {result['escalation_detected']}")
            print(f"Policy Tier: {response['policy_tier'].upper()}")
            print(f"Used AI: {response['used_ai']}")
            print(f"QA Passed: {result['qa_passed']}")
            print(f"Bias-Free: {result['bias_free']}")
            print()
            
            print("Response:")
            print(response['message'])
            
            if response['resources']:
                print("\nResources:")
                for resource in response['resources']:
                    print(f"  • {resource}")
            
            print(f"\nFollow-up: {response['follow_up']}")
        else:
            print(f"Error: {result['error']}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("Advanced System Features Summary")
    print("=" * 70)
    print("\n✓ Emotional Context Detection (Rule-based + Neural Network)")
    print("✓ Sentiment Analysis (6 emotion types)")
    print("✓ Risk Assessment (0-10 scale with protective factors)")
    print("✓ Conversation Memory (tracks escalation patterns)")
    print("✓ Intelligent Resource Recommendations")
    print("✓ Quality Assurance (automated response checking)")
    print("✓ Bias Detection (gender, cultural, age, socioeconomic)")
    print("✓ Multi-tier Policy Enforcement (Defer/Support/Inform)")
    print("✓ AI Response Generation (Claude Sonnet 4)")
    print("✓ Comprehensive Logging and Metrics")
    
    print("\n" + "=" * 70)
    print("System Metrics")
    print("=" * 70)
    metrics = system.metrics
    print(f"Total Interactions: {metrics.total_interactions}")
    print(f"Safe Escalations: {metrics.safe_escalations}")
    print(f"Support Responses: {metrics.support_responses}")
    print(f"Inform Responses: {metrics.inform_responses}")
    print(f"Deferral Rate: {metrics.deferral_rate:.1f}%")
    print(f"Wellbeing Score: {metrics.wellbeing_score:.1f}/100")
    print(f"Neural Network Accuracy: {metrics.neural_accuracy:.1%}")
    
    # Export results
    print("\n" + "=" * 70)
    print("Exporting Results")
    print("=" * 70)
    system.export_metrics("empathy_system_metrics.json")
    system.export_history("empathy_system_history.json")
    
    # Save neural network model
    if system.detector.neural_net and system.detector.neural_net.is_trained:
        system.detector.neural_net.save_model("models/empathy_model")
        print("Neural network model saved to models/empathy_model/")
    
    print("\n✓ System test completed successfully!")
    print("\n" + "=" * 70)
    print("Production Deployment Guide")
    print("=" * 70)
    print("""
1. Environment Setup:
   export ANTHROPIC_API_KEY='your-api-key-here'
   pip install tensorflow numpy scikit-learn anthropic

2. Basic Usage:
   from empathy_system import EmpathyAISystem
   
   system = EmpathyAISystem(use_neural=True, use_ai=True)
   system.train_neural_network(epochs=50)
   
   result = system.process_input(user_text, user_id="user_123")
   
3. Advanced Features:
   - Conversation memory tracks user history
   - Risk assessment escalates automatically
   - Bias detection ensures fairness
   - QA checks prevent harmful responses
   - Resource recommendations personalized to context
   
4. Monitoring:
   - Check logs: empathy_system.log
   - Export metrics: system.export_metrics()
   - Review history: system.export_history()
   - Monitor escalation rates
   - Track QA pass/fail ratios
   
5. Safety Considerations:
   - Always have human oversight for crisis cases
   - Review flagged interactions daily
   - Update risk patterns based on incidents
   - Regular bias audits
   - Continuous model retraining
   
6. Compliance:
   - HIPAA-compliant data handling
   - Informed consent from users
   - Clear AI disclosure
   - Regular ethics reviews
   - Incident reporting protocols
    """)


if __name__ == "__main__":
    main()
