🧠 Homeostatic-Transformer
An experiment with dynamic, bio-inspired homeostatic regulation inside Transformer layers. Instead of treating generation parameters as static constants, this architecture treats stability as a resource.
🔬 Homeostatic Approaches: Autonomous vs. Induced
Approach
Level	Mechanism	Analogy	Project Status
Learned Homeostasis	Architecture Level	Temperature and amnesia are trainable internal states integrated into the forward pass. Model self-regulates.	A healthy brain with perfect self-regulation.	Implemented (homeostatic_transformer.py)
Induced Homeostasis	Inference Level	External override: forcing extreme temperature states to study boundary defense mechanisms.	A brain under chemical stress; observing the breakdown.	Experimental Stress-Test (experiments/)
📊 Core Experiment: Homeostatic-Transformer (Scratch)
We trained a small Homeostatic-Transformer against a Standard Transformer on the TinyStories dataset to observe autonomous state balancing.
Quantitative Metrics
Note: Homeostatic regulation trades raw predictability for structural diversity and repetition suppression.
Model
Diversity ↑	Repeat ↓	Narrative Verbs ↑
Standard Transformer	0.750	0.023	9.2%
Homeostatic-Transformer	0.762	0.018	12.4%
Qualitative Generation Example
Standard: "Once upon a time, there was a little girl named Lily. ... her mom gave her a loud noise on the comet." (Semantic drift, nonsensical ending)
Homeostatic: "Once upon a time, there was a little fish named Tom. Tom loved to play with his friends. ... They played together in the deep sea." (Coherent narrative, stable focus)
💡 Result: Autonomous homeostasis prevents activation explosions, keeping the model inside a productive "cognitive envelope" without hard-coded limits.
💓 Heartbeat & Amnesia Visualization
During text generation, the model dynamically adjusts its internal variables based on tokens processed.
Heartbeat (Upper Plot): Tracks cognitive load (Internal Temperature). A minor spike occurs at the beginning of the sentence ("Once upon a time"), which rapidly self-stabilizes well below the critical failure threshold line (Critical = 2.0).
Amnesia Accumulation (Lower Plot): Shows how layers exponentially decay old, irrelevant context to clear "working memory" as the sequence grows.
🧪 Boundary Stress Test: Induced Loops on Qwen2.5
To understand why the model needs autonomous regulation, we performed a destructive stress-test on Qwen2.5. We artificially forced its external inference temperature into extreme zones to map its defense mechanisms.
Hypothesis: Artificial hyper-activation will force the model to shut down its narrative dynamics to protect coherence.
Method: Prompt: "The dark hall was completely empty. Dust". We compared a baseline control against induced hyper-temperatures (

 and 

).
Stress-Test Results
Mode

Verbs %	Behavioral Response
Control	0.7	17.8%	Coherent Narrative: Standard gothic storytelling.
Induced Stress	1.9	20.3%	Hyper-Arousal / Panic: Compensatory surge in action. Verbs increase; model frantically tries to maintain a story structure under noise.
System Breakdown	2.5	16.7%	Cognitive Phase Shift: Complete collapse of narrative. The model abandons the story entirely and shifts into safe, repetitive structural templates (math formulas, ad banners, test scripts).
⚠️ Conclusion: Under extreme external stress, a model does not degrade smoothly. It undergoes a phase shift in modality. Giving up the narrative to output rigid, predictable templates (like code or ads) is a defensive homeostatic reaction of the token distribution to prevent complete chaos.
🚀 Quick Start
Run everything in your browser — no local GPU needed.
pip install torch datasets tokenizers matplotlib
from homeostatic_transformer import HomeostaticTransformer

model = HomeostaticTransformer(
    vocab_size=5000,
    embed_dim=128,
    num_layers=3,
    critical_temp=2.0
)
# Training loop and dynamic visualizations are available in the notebook
├── homeostatic_transformer.py   # Core architecture (Learned Homeostasis)
├── train_and_evaluate.ipynb     # Scratch training on TinyStories
├── experiments/
│   └── homeostatic_loop/
│       └── run_final.py         # Qwen2.5 induced stress-test script
├── heartbeat.png                # Pulse visualization sample
└── README.md
🧠 Novelty HighlightsLearned Homeostasis: Temperature and context amnesia are not global hyper-parameters; they are dynamic, trainable internal layer states optimized by gradients.Exponential Modulation: Smooth activation dampening instead of aggressive, hard token-filtering thresholds (Top-p/Top-k).Continuous Load History: Internal temperature states persist across token steps, creating a short-term memory of recent "cognitive stress".
