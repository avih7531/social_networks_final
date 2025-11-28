# Predicting Movie Genre Using Actor Co-Appearance Networks

**Name**: Avi Herman

## Abstract (Background)
Film collaboration patterns form structured networks rather than random linkages. Actors frequently reappear alongside others who participate in similar stylistic, thematic, or production niches. Large-scale actor graphs exhibit clustered substructures, and these clusters appear to align with recognizable genre ecosystems such as horror ensembles, action franchises, family-film casts, or dramatic art-house circles. This project evaluates whether genre structure can be inferred **solely from actor co-appearance networks**, without relying on plot descriptions, reviews, text embeddings, or supervised classifiers. If genre can be predicted from network structure alone, this suggests that genre functions not only as a descriptive label but as a **socially emergent formation** within the film industry.

## Introduction (Hypothesis, Logic, Falsifiability)
**Hypothesis.** An undirected actor network where nodes represent actors and edges represent co-appearances will exhibit **community structure** with high modularity, and these communities will correspond to film genres at rates above random assignment.  
**Reasoning.** Actors specializing in similar film types tend to co-appear repeatedly, producing clustered subgraphs. If these clusters reflect genre ecosystems, community detection should identify them.  
**Falsifiability.** The hypothesis is rejected if:
• modularity remains low across all genre subsets  
• detected communities lack dominant genre composition  
• genre prediction accuracy approaches random chance  

## Definitions
• **Actor co-appearance network.** Undirected weighted graph \(G=(V,E)\) where each \(v \in V\) is an actor and an edge \(e_{ij}\) exists if actors \(i\) and \(j\) appear in the same film; weight \(w_{ij}\) increases with repeated co-appearances.  
• **Modularity.** Quality function measuring excess within-community connectivity relative to a null model:  
\[
Q = \frac{1}{2m} \sum_{i,j}\Big(A_{ij} - \frac{k_i k_j}{2m}\Big)\,\delta(c_i,c_j),
\]
where \(A\) is adjacency, \(k_i\) node degree or strength, \(2m=\sum_{ij}A_{ij}\), and \(c_i\) community labels.  
• **Louvain algorithm.** Greedy maximization of \(Q\) producing non-overlapping communities efficiently on large graphs.  
• **Genre purity.** For a community with genre-count vector \((g_1,g_2,\dots)\), purity \(P = g_1 / \sum g_i\).  
• **Genre prediction accuracy.** The proportion of actor-genre assignments correctly inferred by assigning each community its dominant genre.

## MVE (Data, Sample Size, Tipping Point)
**Data.** IMDb title.basics.tsv and title.principals.tsv restricted to films and actors, including both **single-genre** and **multi-genre** movies to observe overlap effects.  
**Sample size.** Up to \(5000\) films per configuration, producing actor networks ranging from approximately \(12{,}000\) to \(14{,}000\) nodes and \(30{,}000\) to \(36{,}000\) edges.  
**Tipping point.** The experiment varies the number of included genres from \(3\) to \(12\), observing when \(Q\), purity, and accuracy stabilize or decay, indicating the limits of genre recoverability from collaboration networks.

## Analysis (Statistical Analysis, Scaling or Pivoting)
Two graphs are constructed for each experiment:  
• **unweighted:** each co-appearance increments \(w_{ij}\) by \(1\)  
• **weighted:** \(w_{ij}\) scaled by \(1/|G_\text{movie}|\) where \(|G_\text{movie}|\) is the number of genres assigned to that film  
Louvain community detection yields modularity \(Q\), community counts, purity levels, and prediction accuracy. Comparative analysis examines whether structure persists, weakens, or fragments as the number of genres increases. Scaling pathways include enlarging samples, isolating niche clusters such as horror or war, and segmenting data temporally to observe historical drift.

## Conclusion
Actor collaboration networks contain sufficient structural regularity to recover genre-aligned communities at rates meaningfully above random prediction. As the number of genres increases, overlap reduces purity and accuracy, indicating that genre ecosystems intersect rather than forming perfectly isolated partitions. Results support the interpretation of genre not only as a content category but as an **emergent network community** grounded in the social structure of film production.
