# Code organization

### A few pointers

-   [Markov-LLM](Markov-LLM) contains the full transformer model for binary first-order Markov sources.
-   [Markov-LLM-k](Markov-LLM-k) contains the full transformer model for binary k-order Markov sources.
-   [Markov-LLM-m](Markov-LLM-m) contains the full transformer model for Markov sources with arbitrary vocabulary size.
-   [Markov-LLM-depth](Markov-LLM-depth) contains code to run experiments that also save parameter weights across iterations.
-   [Markov-Simple](Markov-Simple) contains a simplified transformer model for first-order Markov sources without layer norm.
-   [Markov-RPE](Markov-RPE) contains the full transformer model with relative positional embeddings.
-   [Markov-Fixed](Markov-Fixed) contains a three-parameter simple architecture that mimicks a transformer model with rank-one parameter initialization.

The script to run the experiments is in src/main.py.
