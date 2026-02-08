"""
Temporal Clustering Analysis for Naloxone Mentions
Uses UMAP + HDBSCAN with sentence-transformers embeddings
Generates interactive Plotly visualizations with quarterly aggregation
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
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

print("\nApplying HDBSCAN clustering...")
# Clustering on the full embedding space (not reduced) for better results
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=25,
    min_samples=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)
cluster_labels = clusterer.fit_predict(embeddings)

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
print(f"\nClustering Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

# Create cluster labels (handle noise)
df_sample['cluster_label'] = df_sample['cluster'].apply(
    lambda x: f'Cluster {x}' if x >= 0 else 'Noise'
)

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
fig1.write_html('naloxone_clustering_2d.html')
print("✓ Saved: naloxone_clustering_2d.html")

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
fig2.write_html('naloxone_clustering_2d_time.html')
print("✓ Saved: naloxone_clustering_2d_time.html")

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
fig3.write_html('naloxone_clustering_3d_time.html')
print("✓ Saved: naloxone_clustering_3d_time.html")

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
fig4.update_traces(marker=dict(size=4, opacity=0.6))
fig4.write_html('naloxone_clustering_3d_umap.html')
print("✓ Saved: naloxone_clustering_3d_umap.html")

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
fig5.write_html('naloxone_cluster_evolution.html')
print("\n✓ Saved: naloxone_cluster_evolution.html")

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

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. naloxone_clustering_2d.html - 2D UMAP colored by cluster")
print("  2. naloxone_clustering_2d_time.html - 2D UMAP colored by time")
print("  3. naloxone_clustering_3d_time.html - 3D with time as Z-axis")
print("  4. naloxone_clustering_3d_umap.html - 3D UMAP visualization")
print("  5. naloxone_cluster_evolution.html - Cluster trends over time")
print("\nOpen these HTML files in your browser to explore interactively!")
