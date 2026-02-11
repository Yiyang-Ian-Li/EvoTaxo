"""
Temporal Clustering Analysis for Naloxone Mentions
Uses UMAP + multiple clustering algorithms (HDBSCAN vs KMeans)
Generates interactive Plotly visualizations with quarterly aggregation
Saves organized output in visualizations/ directory
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = 'visualizations'
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}/")

print("\nLoading data...")
# Load the dataset
df = pd.read_csv('naloxone_mentions.csv')

# Convert to datetime
df['created_dt'] = pd.to_datetime(df['created_dt'])

# Create quarter column
df['year'] = df['created_dt'].dt.year
df['quarter'] = df['created_dt'].dt.quarter
df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

# Sample for faster processing (can adjust or remove)
print(f"Total records: {len(df)}")
sample_size = min(10000, len(df))  # Use up to 10k posts for speed
df_sample = df.sample(n=sample_size, random_state=42).copy()
print(f"Using {len(df_sample)} sampled records")

# Clean text
print("\nCleaning text...")
df_sample['text_clean'] = df_sample['text'].fillna('').astype(str)
df_sample = df_sample[df_sample['text_clean'].str.len() > 20]  # Filter very short texts
print(f"After filtering: {len(df_sample)} records")

print("\nGenerating embeddings with sentence-transformers...")
print("Loading model 'all-MiniLM-L6-v2'...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
texts = df_sample['text_clean'].tolist()
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
print(f"Embeddings shape: {embeddings.shape}")

print("\nApplying UMAP dimensionality reduction...")
# UMAP for 2D visualization
umap_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
embeddings_2d = umap_2d.fit_transform(embeddings)

# UMAP for 3D (for 3D viz)
umap_3d = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42
)
embeddings_3d = umap_3d.fit_transform(embeddings)

print("\n" + "="*70)
print("COMPARING CLUSTERING ALGORITHMS")
print("="*70)

# Method 1: HDBSCAN with strict parameters
print("\n1. HDBSCAN (strict)...")
hdbscan_strict = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
labels_strict = hdbscan_strict.fit_predict(embeddings)
n_clusters_strict = len(set(labels_strict)) - (1 if -1 in labels_strict else 0)
n_noise_strict = list(labels_strict).count(-1)
noise_pct_strict = n_noise_strict / len(labels_strict) * 100
print(f"   Clusters: {n_clusters_strict}, Noise: {noise_pct_strict:.1f}%")

# Method 2: HDBSCAN with lenient parameters
print("\n2. HDBSCAN (lenient)...")
hdbscan_lenient = hdbscan.HDBSCAN(
    min_cluster_size=15,
    min_samples=3,
    metric='euclidean',
    cluster_selection_method='eom',
    cluster_selection_epsilon=0.5,
    prediction_data=True
)
labels_lenient = hdbscan_lenient.fit_predict(embeddings)
n_clusters_lenient = len(set(labels_lenient)) - (1 if -1 in labels_lenient else 0)
n_noise_lenient = list(labels_lenient).count(-1)
noise_pct_lenient = n_noise_lenient / len(labels_lenient) * 100
print(f"   Clusters: {n_clusters_lenient}, Noise: {noise_pct_lenient:.1f}%")

# Method 3: HDBSCAN on UMAP embeddings (lower dimensional)
print("\n3. HDBSCAN on UMAP 2D...")
hdbscan_umap = hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
labels_umap = hdbscan_umap.fit_predict(embeddings_2d)
n_clusters_umap = len(set(labels_umap)) - (1 if -1 in labels_umap else 0)
n_noise_umap = list(labels_umap).count(-1)
noise_pct_umap = n_noise_umap / len(labels_umap) * 100
print(f"   Clusters: {n_clusters_umap}, Noise: {noise_pct_umap:.1f}%")

# Method 4: KMeans with different k values
print("\n4. KMeans (trying multiple k)...")
best_k = None
best_kmeans_score = -1
best_kmeans_labels = None

for k in [5, 7, 10, 12, 15]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = kmeans.fit_predict(embeddings)
    sil_k = silhouette_score(embeddings, labels_k)
    print(f"   k={k:2d}: silhouette={sil_k:.3f}")
    
    if sil_k > best_kmeans_score:
        best_kmeans_score = sil_k
        best_k = k
        best_kmeans_labels = labels_k

print(f"   Best: k={best_k} with silhouette={best_kmeans_score:.3f}")

# Calculate silhouette scores for HDBSCAN methods (excluding noise)
sil_scores = {}
for name, labels in [('strict', labels_strict), ('lenient', labels_lenient), ('umap', labels_umap)]:
    non_noise_mask = labels >= 0
    if non_noise_mask.sum() > 10 and len(set(labels[non_noise_mask])) > 1:
        sil_scores[name] = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
    else:
        sil_scores[name] = -1

# Selection logic
print("\n" + "="*70)
print("ALGORITHM SELECTION")
print("="*70)

methods = {
    'HDBSCAN-strict': (labels_strict, n_clusters_strict, noise_pct_strict, sil_scores['strict']),
    'HDBSCAN-lenient': (labels_lenient, n_clusters_lenient, noise_pct_lenient, sil_scores['lenient']),
    'HDBSCAN-UMAP': (labels_umap, n_clusters_umap, noise_pct_umap, sil_scores['umap']),
    f'KMeans-{best_k}': (best_kmeans_labels, best_k, 0, best_kmeans_score)
}

# Choose best method
# Prefer methods with <75% noise and >3 clusters
viable_methods = []
for name, (labels, n_clust, noise_pct, sil) in methods.items():
    if noise_pct < 75 and n_clust >= 3:
        viable_methods.append((name, labels, n_clust, noise_pct, sil))

if not viable_methods:
    # Fall back to KMeans if all HDBSCAN have too much noise
    print(f"✓ Using KMeans (k={best_k}) - all HDBSCAN methods have >75% noise")
    cluster_labels = best_kmeans_labels
    clustering_method = f'KMeans (k={best_k})'
    selected_silhouette = best_kmeans_score
else:
    # Choose method with best silhouette score among viable options
    viable_methods.sort(key=lambda x: x[4], reverse=True)
    best_method = viable_methods[0]
    clustering_method = best_method[0]
    cluster_labels = best_method[1]
    selected_silhouette = best_method[4]
    print(f"✓ Using {clustering_method}")
    print(f"  Clusters: {best_method[2]}, Noise: {best_method[3]:.1f}%, Silhouette: {best_method[4]:.3f}")

# Add results to dataframe
df_sample['cluster'] = cluster_labels
df_sample['umap_x'] = embeddings_2d[:, 0]
df_sample['umap_y'] = embeddings_2d[:, 1]
df_sample['umap_3d_x'] = embeddings_3d[:, 0]
df_sample['umap_3d_y'] = embeddings_3d[:, 1]
df_sample['umap_3d_z'] = embeddings_3d[:, 2]

# Convert time to numeric (quarters since start)
min_date = df_sample['created_dt'].min()
df_sample['time_numeric'] = ((df_sample['created_dt'] - min_date).dt.days / 91.25).round(0)  # Approximate quarters

# Cluster statistics
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)
noise_pct_final = n_noise/len(cluster_labels)*100

print("\n" + "="*70)
print(f"FINAL CLUSTERING: {clustering_method}")
print("="*70)
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({noise_pct_final:.1f}%)")
print(f"  Silhouette score: {selected_silhouette:.3f}")

# Create cluster labels (handle noise)
df_sample['cluster_label'] = df_sample['cluster'].apply(
    lambda x: f'Cluster {x}' if x >= 0 else 'Noise'
)

# Cluster size distribution
print("\nCluster size distribution:")
for cid in sorted(df_sample['cluster'].unique()):
    count = (df_sample['cluster'] == cid).sum()
    pct = count / len(df_sample) * 100
    if cid >= 0:
        print(f"  Cluster {cid}: {count:4d} posts ({pct:5.1f}%)")
    else:
        print(f"  Noise:      {count:4d} posts ({pct:5.1f}%)")

print("\n" + "="*60)
print("VISUALIZATION 1: 2D UMAP Clustering by Quarter")
print("="*60)

# 2D visualization colored by cluster
fig1 = px.scatter(
    df_sample,
    x='umap_x',
    y='umap_y',
    color='cluster_label',
    hover_data={
        'year_quarter': True,
        'subreddit': True,
        'text_clean': True,
        'umap_x': False,
        'umap_y': False
    },
    title='UMAP Projection of Naloxone Reddit Posts (Colored by Cluster)',
    labels={'umap_x': 'UMAP Dimension 1', 'umap_y': 'UMAP Dimension 2'},
    width=1000,
    height=700
)
fig1.update_traces(marker=dict(size=5, opacity=0.6))
output_path = os.path.join(output_dir, 'clustering_2d_by_cluster.html')
fig1.write_html(output_path)
print(f"✓ Saved: {output_path}")

print("\n" + "="*60)
print("VISUALIZATION 2: 2D UMAP by Time Period")
print("="*60)

# 2D visualization colored by time
fig2 = px.scatter(
    df_sample,
    x='umap_x',
    y='umap_y',
    color='year_quarter',
    hover_data={
        'cluster_label': True,
        'subreddit': True,
        'text_clean': True,
        'umap_x': False,
        'umap_y': False
    },
    title='UMAP Projection Colored by Time Period',
    labels={'umap_x': 'UMAP Dimension 1', 'umap_y': 'UMAP Dimension 2'},
    width=1000,
    height=700
)
fig2.update_traces(marker=dict(size=5, opacity=0.6))
output_path = os.path.join(output_dir, 'clustering_2d_by_time.html')
fig2.write_html(output_path)
print(f"✓ Saved: {output_path}")

print("\n" + "="*60)
print("VISUALIZATION 3: 3D Visualization with Time as Z-axis")
print("="*60)

# 3D visualization: UMAP dims as X,Y and time as Z
fig3 = go.Figure()

# Add traces for each cluster
for cluster_id in sorted(df_sample['cluster'].unique()):
    if cluster_id >= 0:
        cluster_data = df_sample[df_sample['cluster'] == cluster_id]
        fig3.add_trace(go.Scatter3d(
            x=cluster_data['umap_x'],
            y=cluster_data['umap_y'],
            z=cluster_data['time_numeric'],
            mode='markers',
            name=f'Cluster {cluster_id}',
            marker=dict(size=4, opacity=0.6),
            text=cluster_data['text_clean'].str[:200],
            hovertemplate='<b>Cluster %{fullData.name}</b><br>' +
                          'Time: %{z:.0f} quarters<br>' +
                          '%{text}<extra></extra>'
        ))

# Add noise as separate trace
noise_data = df_sample[df_sample['cluster'] == -1]
if len(noise_data) > 0:
    fig3.add_trace(go.Scatter3d(
        x=noise_data['umap_x'],
        y=noise_data['umap_y'],
        z=noise_data['time_numeric'],
        mode='markers',
        name='Noise',
        marker=dict(size=3, opacity=0.3, color='gray'),
        text=noise_data['text_clean'].str[:200],
        hovertemplate='<b>Noise</b><br>Time: %{z:.0f} quarters<br>%{text}<extra></extra>'
    ))

fig3.update_layout(
    title='3D Temporal Clustering: UMAP Space with Time Axis',
    scene=dict(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        zaxis_title='Time (Quarters since start)',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
    ),
    width=1100,
    height=800
)
output_path = os.path.join(output_dir, 'clustering_3d_time_axis.html')
fig3.write_html(output_path)
print(f"✓ Saved: {output_path}")

print("\n" + "="*60)
print("VISUALIZATION 4: 3D UMAP Visualization")
print("="*60)

# Alternative 3D: Using all 3 UMAP dimensions, color by cluster
fig4 = px.scatter_3d(
    df_sample,
    x='umap_3d_x',
    y='umap_3d_y',
    z='umap_3d_z',
    color='cluster_label',
    hover_data={
        'year_quarter': True,
        'subreddit': True,
        'text_clean': True,
        'umap_3d_x': False,
        'umap_3d_y': False,
        'umap_3d_z': False
    },
    title='3D UMAP Projection (All UMAP Dimensions)',
    labels={
        'umap_3d_x': 'UMAP Dimension 1',
        'umap_3d_y': 'UMAP Dimension 2',
        'umap_3d_z': 'UMAP Dimension 3'
    },
    width=1100,
    height=800
)
output_path = os.path.join(output_dir, 'clustering_3d_umap.html')
fig4.write_html(output_path)
print(f"✓ Saved: {output_path}")

print("\n" + "="*60)
print("TEMPORAL CLUSTER ANALYSIS")
print("="*60)

# Analyze cluster distribution over time
cluster_time = df_sample[df_sample['cluster'] >= 0].groupby(['year_quarter', 'cluster']).size().reset_index(name='count')
cluster_time_pivot = cluster_time.pivot(index='year_quarter', columns='cluster', values='count').fillna(0)

print("\nCluster distribution by quarter:")
print(cluster_time_pivot)

# Plot cluster evolution over time
fig5 = go.Figure()
for col in cluster_time_pivot.columns:
    fig5.add_trace(go.Scatter(
        x=cluster_time_pivot.index,
        y=cluster_time_pivot[col],
        mode='lines+markers',
        name=f'Cluster {col}',
        line=dict(width=2),
        marker=dict(size=6)
    ))

fig5.update_layout(
    title='Cluster Size Evolution Over Time',
    xaxis_title='Quarter',
    yaxis_title='Number of Posts',
    hovermode='x unified',
    width=1200,
    height=600
)
fig5.update_xaxes(tickangle=45)
output_path = os.path.join(output_dir, 'cluster_evolution_timeline.html')
fig5.write_html(output_path)
print(f"\n✓ Saved: {output_path}")

# Sample posts from each cluster
print("\n" + "="*60)
print("CLUSTER EXAMPLES")
print("="*60)

for cluster_id in sorted(df_sample['cluster'].unique()):
    if cluster_id >= 0:
        cluster_posts = df_sample[df_sample['cluster'] == cluster_id]
        print(f"\n--- Cluster {cluster_id} ({len(cluster_posts)} posts) ---")
        print(f"Subreddits: {cluster_posts['subreddit'].value_counts().head(3).to_dict()}")
        print(f"Time range: {cluster_posts['year_quarter'].min()} to {cluster_posts['year_quarter'].max()}")
        print("\nSample posts:")
        for idx, (_, row) in enumerate(cluster_posts.head(3).iterrows(), 1):
            text_preview = row['text_clean'][:150].replace('\n', ' ')
            print(f"  {idx}. [{row['year_quarter']}] {text_preview}...")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\n✓ Method: {clustering_method}")
print(f"✓ Clusters: {n_clusters}, Noise: {noise_pct_final:.1f}%")
print(f"✓ Quality: Silhouette = {selected_silhouette:.3f}")
print(f"\nAll visualizations saved to: {output_dir}/")
print("\nGenerated files:")
print("  1. clustering_2d_by_cluster.html - 2D UMAP colored by cluster")
print("  2. clustering_2d_by_time.html - 2D UMAP colored by time period")
print("  3. clustering_3d_time_axis.html - 3D with time as Z-axis")
print("  4. clustering_3d_umap.html - 3D UMAP visualization")
print("  5. cluster_evolution_timelinee.html - 3D with time as Z-axis")
print("  4. naloxone_clustering_3d_umap.html - 3D UMAP visualization")
print("  5. naloxone_cluster_evolution.html - Cluster trends over time")
print("\nOpen these HTML files in your browser to explore interactively!")
