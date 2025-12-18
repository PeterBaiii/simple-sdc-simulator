[TOC]

# Logs in Experiments

## NaN in Bounds Checking

* Problem came from the product of '0' and 'inf'
* Intermediate result is highly related with value of embedding
    * If the vocab size is small while embedding dimension is large, then the embedding vector will be sparse

**Two problems:**
* By derivation, top2 a_j is strictly the first two maximal value, even if they are the same
* When introduce causal mask into attention, the bounds should be rewritten since 'gamma' is different

**Trials**
* Top 2 different value
* Bounds even for attention with masks

**Further Steps**
* Modularization
* Run in different random seed
* Different injection position
* More complete experiments
* Different top2 score matrix values
* Derivation with masked attention
* New bounds
* Profiling the checker's performance