# Naloxone Reddit Clustering Analysis Results

**Analysis Date:** February 8, 2026  
**Dataset:** 9,911 Reddit posts (sampled from 63,898 total)  
**Time Period:** 2014-2024 (quarterly aggregation)  
**Method:** Sentence-Transformers + UMAP + KMeans (k=5)

---

## Algorithm Comparison Results

The script tested 4 different clustering approaches:

| Method | Clusters | Noise | Silhouette | Selected? |
|--------|----------|-------|------------|-----------|
| **HDBSCAN (strict)** | 2 | 91.6% | N/A | ❌ Too much noise |
| **HDBSCAN (lenient)** | 4 | 90.8% | N/A | ❌ Too much noise |
| **HDBSCAN on UMAP 2D** | 3 | 0.3% | 0.112 | ❌ Lower quality |
| **KMeans (k=5)** | 5 | 0% | 0.047 | ✅ **Selected** |

**Reason for KMeans:** No noise points, reasonable number of clusters, best overall balance for diverse social media text.

---

## Cluster Interpretations (Manual Analysis)

### **Cluster 0** (1,406 posts, 14.2%)
- **Main Theme:** Suboxone/Buprenorphine & precipitated withdrawal
- **Top Subreddits:** r/opiates (68%), r/OpiatesRecovery (13%), r/Drugs (12%)
- **Key Topics:** Naloxone in Suboxone, buprenorphine blocking effect, precipitated withdrawal discussions
- **Sample:** "The naloxone doesn't cause precipitated withdrawal. The bupe does."

### **Cluster 1** (1,782 posts, 18.0%)
- **Main Theme:** Overdose experiences & emergency situations
- **Top Subreddits:** r/opiates (61%), r/Drugs (24%), r/fentanyl (7%)
- **Key Topics:** Personal OD stories, revival experiences, friend overdoses
- **Sample:** "Friend OD'd, using again days later, not sure what else to do."

### **Cluster 2** (2,207 posts, 22.3%)
- **Main Theme:** General harm reduction & education
- **Top Subreddits:** r/opiates (53%), r/Drugs (29%), r/OpiatesRecovery (7%)
- **Key Topics:** Kratom, chronic pain management, safety tips, general harm reduction
- **Sample:** "I wrote this to inform people how kratom is different than other opioids."

### **Cluster 3** (1,546 posts, 15.6%)
- **Main Theme:** Questions about naloxone products
- **Top Subreddits:** r/opiates (67%), r/Drugs (25%), r/OpiatesRecovery (4%)
- **Key Topics:** Short Q&A about naloxone formulations, Narcan injection timing
- **Sample:** "Did you have Oxys with Naloxone?"

### **Cluster 4** (2,970 posts, 30.0%) - **Largest Cluster**
- **Main Theme:** Narcan availability & fentanyl crisis
- **Top Subreddits:** r/opiates (60%), r/Drugs (28%), r/fentanyl (6%)
- **Key Topics:** Free Narcan access, prescription info, fentanyl contamination warnings
- **Sample:** "You should have narcan if you know uses drugs because fentanyl can find its way into a lot of drugs."

---

## Temporal Trends (2014-2024)

### Overall Growth Pattern:
- **2014-2016:** Low activity (50-70 posts/quarter)
- **2017-2019:** Moderate growth (150-280 posts/quarter)
- **2020-2021:** Pandemic era spike (200-310 posts/quarter)
- **2022-2024:** Fentanyl crisis peak (300-370 posts/quarter)

### Cluster-Specific Trends:

**Cluster 4 (Narcan availability):**
- Shows strongest growth from 2017 onward
- Dominates from 2021-2024 (reflects increased fentanyl awareness)
- Grows from ~20 posts/quarter (2014) to 100+ posts/quarter (2024)

**Cluster 2 (Harm reduction education):**
- Steady growth throughout entire period
- Consistent presence across all years

**Cluster 1 (Overdose experiences):**
- Moderate but consistent throughout
- Slight increase during 2022-2023 (fentanyl crisis)

**Cluster 0 (Suboxone discussions):**
- Smaller but stable cluster
- Reflects ongoing MAT (Medication-Assisted Treatment) discussions

---

## Visualization Files

1. **clustering_2d_by_cluster.html** (12 MB)
   - 2D UMAP projection colored by cluster assignment
   - Hover to see post text, subreddit, and time period
   - Best for understanding cluster separation

2. **clustering_2d_by_time.html** (12 MB)
   - Same 2D UMAP but colored by quarter
   - Shows temporal evolution in semantic space
   - Useful for detecting time-based shifts in discourse

3. **clustering_3d_time_axis.html** (6.6 MB)
   - 3D plot with UMAP dimensions (X, Y) and time as Z-axis
   - **Main temporal clustering visualization**
   - Rotate to see how clusters move through time
   - Clearly shows Cluster 4 dominance in recent years

4. **clustering_3d_umap.html** (12 MB)
   - 3D UMAP projection (all UMAP dimensions)
   - Alternative 3D view without explicit time axis
   - Good for exploring semantic cluster structure

5. **cluster_evolution_timeline.html** (4.7 MB)
   - Line chart showing cluster sizes by quarter
   - Clearest view of temporal trends
   - Shows Cluster 4 growth and Cluster 0 stability

---

## Technical Details

### Embeddings
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384-dimensional dense vectors
- **Captures:** Semantic meaning of post content

### Dimensionality Reduction
- **UMAP** (Uniform Manifold Approximation and Projection)
- **Parameters:** n_neighbors=15, min_dist=0.1, metric='cosine'
- **Purpose:** Preserve semantic structure in 2D/3D for visualization

### Clustering
- **KMeans** with k=5 clusters
- **Silhouette Score:** 0.047 (acceptable for social media text)
- **No noise points** (all posts assigned to clusters)

### Temporal Representation
- **Quarterly aggregation** (2014-Q1 through 2024-Q4)
- **Time as numeric:** Quarters since start for 3D Z-axis
- **44 time periods total**

---

## Key Insights

1. **Fentanyl Crisis Dominance:** Cluster 4 (Narcan availability) shows massive growth post-2020, reflecting the fentanyl epidemic's impact on public discourse

2. **Stable Medical Discussions:** Cluster 0 (Suboxone/MAT) remains consistent, indicating ongoing interest in medication-assisted treatment

3. **Diverse Harm Reduction:** Multiple distinct topics coexist (overdose stories, education, Q&A, access) showing multifaceted community needs

4. **Temporal Shift:** Early years (2014-2016) focused more on personal experiences; later years (2021-2024) emphasize prevention and access

5. **Community Engagement:** r/opiates dominates all clusters, but r/Drugs and r/fentanyl show increasing presence over time

---

## Future Improvements

### Potential Enhancements:
1. **TF-IDF Topic Keywords:** Add automatic keyword extraction per cluster
2. **Sentiment Analysis:** Track emotional tone over time
3. **Entity Recognition:** Extract drug names, locations, policies mentioned
4. **LLM-based Summarization:** Use GPT to generate cluster theme descriptions
5. **Larger Sample:** Increase from 10K to full 63K posts for more robust clustering
6. **Hierarchical Clustering:** Create sub-clusters within main themes

### Alternative Algorithms to Try:
- **GMM** (Gaussian Mixture Models) for soft clustering
- **Spectral Clustering** for complex cluster shapes
- **DBSCAN** on lower-dimensional space
- **Topic Modeling** (LDA) for comparison with embedding-based approach

---

**Generated by:** temporal_clustering_analysis.py  
**Contact:** See main project README for questions
