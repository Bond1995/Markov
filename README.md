# Code organization

### A few pointers

-   [Markov-LLM](Markov-LLM) contains the full transformer model for first-order Markov sources.
-   [Markov-LLM-k](Markov-LLM-k) contains the full transformer model for k-order Markov sources of the form described in Sec. 4.
-   [Markov-LLM-m](Markov-LLM-m) contains the full transformer model for k-order Markov sources with arbitrary vocabulary size.
-   [Markov-Simple](Markov-Simple) contains the simplified transformer model for first-order Markov sources used for the experiments in Sec. 3.

The script to run the experiments is in src/main.py.
