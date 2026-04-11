# STACK.md — Technology Decisions

This file documents the technology choices for the Gut Feeling Parameter research repository and the reasoning behind each.

| Technology | Version | Why Chosen | Alternatives Considered |
|---|---|---|---|
| Python | 3.11+ | Stable release with good typing support and performance improvements over 3.10; async available if needed for future data pipeline work | 3.10 (older, missing some typing improvements); 3.12 (too new for stable support across some scientific libs at project start) |
| PyTorch | 2.3+ | Native REINFORCE implementation via `torch.distributions`; LSTM and gradient tools built-in; wide adoption in research means examples and debugging resources are plentiful | JAX (steeper learning curve, functional paradigm harder to onboard onto for collaborators); TensorFlow (less Pythonic, heavier ecosystem for this scale of experiment) |
| scikit-learn | 1.4+ | Linear probe for evaluating learned representations; t-SNE for hidden state visualization; stratified splits for balanced evaluation; well-tested and stable API | Manual implementation of linear probe and t-SNE (unnecessary given sklearn's coverage and reliability) |
| transformers (HuggingFace) | 4.40+ | GPT-2 access for Phase 3 LM application; tokenizer and model loading handled cleanly; active development aligned with current LM research | MLC (for edge/mobile deployment — possible future consideration if GFP is applied on-device); LiteRT (same — edge-focused, not relevant at current research stage) |
| scipy | 1.13+ | Welch's t-test (`scipy.stats.ttest_ind`) for comparing config performance across seeds; Cohen's d effect size computation; clean interface for standard statistical tests | statsmodels (more complex API for the same tests we need; better suited for regression/GLM work not currently in scope); manual implementation (error-prone for edge cases in Welch's test) |
| matplotlib | 3.8+ | Standard research plotting; full control over figure layout and styling; outputs publication-ready static figures | plotly (interactive but heavier; interactivity not needed for static paper figures; harder to style consistently) |
| seaborn | 0.13+ | Cleaner defaults for statistical plots (violin plots, box plots, heatmaps); integrates directly with matplotlib axes | plotly (same reasoning as above) |
| numpy | 1.26+ | Core array operations; signal generation for the 1-D environment; well-integrated with PyTorch via zero-copy tensor bridging | cupy (GPU-accelerated numpy — overkill for the data volumes in this experiment; would add a non-trivial install dependency for CPU runs) |
