# CrewAI Agent Setup 🤖

## What is CrewAI?

CrewAI is a framework for orchestrating role-playing AI agents. It's perfect for complex tasks like overfitting analysis because it uses multiple specialized agents working together.

## Installation

```bash
cd ml-visualizer/backend

# Install CrewAI and dependencies
pip install crewai>=0.1.0
pip install langchain-openai>=0.0.5

# Or install all requirements
pip install -r requirements.txt
```

## How It Works

The CrewAI agent system uses **3 specialized agents**:

### 1. 🔍 Diagnostic Agent
- **Role**: ML Diagnostics Expert
- **Task**: Analyze model metrics and identify overfitting
- **Output**: Severity assessment and root cause analysis

### 2. 💡 Solution Agent
- **Role**: ML Solutions Architect  
- **Task**: Recommend actionable solutions
- **Output**: Prioritized recommendations with impact estimates

### 3. 💻 Code Agent
- **Role**: ML Code Generator
- **Task**: Generate production-ready code examples
- **Output**: sklearn code snippets with explanations

## Advantages over Simple OpenAI

### CrewAI Benefits:
- ✅ **Structured workflow** - Agents work in sequence
- ✅ **Specialized expertise** - Each agent has a specific role
- ✅ **Better quality** - Multiple perspectives on the problem
- ✅ **More detailed** - Comprehensive analysis from different angles
- ✅ **Production-ready** - Built for enterprise applications

### Simple OpenAI:
- ✅ **Faster** - Single API call
- ✅ **Cheaper** - One model invocation
- ✅ **Simpler** - Less dependencies
- ✅ **Fallback** - Works if CrewAI fails

## Usage

The system automatically uses CrewAI if available, otherwise falls back to simple OpenAI:

```python
# Backend automatically chooses:
try:
    from crewai_overfitting_agent import CrewAIOverfittingAgent
    # Uses CrewAI (3 agents working together)
except ImportError:
    from ai_overfitting_agent import AIOverfittingAgent
    # Uses simple OpenAI (single call)
```

## Testing

```bash
# Test CrewAI installation
python -c "import crewai; print('CrewAI installed:', crewai.__version__)"

# Test the agent
python backend/test_openai.py
```

## Cost Comparison

### CrewAI (3 agents):
- **Cost**: ~$0.003-0.009 per analysis
- **Time**: 10-20 seconds
- **Quality**: Excellent (multiple perspectives)

### Simple OpenAI (1 call):
- **Cost**: ~$0.001-0.003 per analysis
- **Time**: 3-5 seconds
- **Quality**: Good (single perspective)

## Troubleshooting

### "ModuleNotFoundError: No module named 'crewai'"
```bash
pip install crewai langchain-openai
```

### "CrewAI not available, using simple OpenAI agent"
This is normal! The system falls back to simple OpenAI if CrewAI isn't installed.

### CrewAI is slow
CrewAI makes multiple LLM calls (one per agent), so it's slower but more thorough.

### Want to force simple OpenAI?
Comment out the CrewAI import in `app.py`:
```python
# from crewai_overfitting_agent import CrewAIOverfittingAgent
from ai_overfitting_agent import AIOverfittingAgent
```

## Architecture

```
User Request
    ↓
Backend API
    ↓
Try CrewAI → Success → 3 Agents → Comprehensive Analysis
    ↓
Fallback to Simple OpenAI → 1 Call → Quick Analysis
    ↓
Return Results
```

## Agent Workflow

```
1. Diagnostic Agent
   ↓ (analyzes metrics)
   Severity: HIGH, Gap: 0.15
   
2. Solution Agent  
   ↓ (recommends fixes)
   Top 5 solutions with priorities
   
3. Code Agent
   ↓ (generates code)
   sklearn examples for each solution
   
→ Combined Report
```

## Example Output

### CrewAI Output:
```
🔍 DIAGNOSTIC AGENT:
Severity: HIGH overfitting detected
Root cause: Model complexity too high for dataset size
Impact: 15% performance gap between train and test

💡 SOLUTION AGENT:
1. Reduce max_depth to 5 (HIGH priority, HIGH impact)
2. Add min_samples_split=10 (HIGH priority, MEDIUM impact)
3. Use cross-validation (MEDIUM priority, MEDIUM impact)

💻 CODE AGENT:
# Solution 1: Reduce complexity
model = RandomForestClassifier(
    max_depth=5,  # Reduced from default
    n_estimators=50,  # Fewer trees
    random_state=42
)
```

## When to Use Each

### Use CrewAI when:
- You want comprehensive analysis
- Quality > Speed
- You have time for detailed recommendations
- You're making important decisions

### Use Simple OpenAI when:
- You want quick feedback
- Speed > Depth
- You're doing rapid iteration
- You want lower costs

## Future Enhancements

- [ ] Add memory to agents (learn from past analyses)
- [ ] Add tool usage (agents can run code)
- [ ] Add human-in-the-loop (ask for clarification)
- [ ] Add custom agent personalities
- [ ] Add multi-model support (GPT-4, Claude, etc.)

---

**Ready to use CrewAI?** Install it and restart your backend! 🚀
