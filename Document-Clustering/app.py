import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA, NMF
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.manifold import TSNE
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO
import os
import time

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Set page config
st.set_page_config(
    page_title="Document Clustering Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stSelectbox>div>div>select {
        background-color: #f8f9fa;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words 
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def plot_wordcloud(text, title=None):
    wordcloud = WordCloud(width=800, height=400,
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=14)
    st.pyplot(plt)

def plot_topic_heatmap(components, feature_names, n_words=100):
    plt.figure(figsize=(12, 8))
    sns.heatmap(components[:, :n_words],
                cmap="YlOrRd",
                yticklabels=[f"Topic {i}" for i in range(components.shape[0])],
                xticklabels=feature_names[:n_words])
    plt.title("Topic-Word Heatmap (First 100 Words)")
    plt.xlabel("Words")
    plt.ylabel("Topics")
    plt.xticks(rotation=90)
    st.pyplot(plt)

def plot_confusion_matrix(true_labels, pred_labels, title):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    st.pyplot(plt)

def get_table_download_link(df, filename="results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

def load_sample_data(sample_size):
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            newsgroups = fetch_20newsgroups(subset='all', 
                                          remove=('headers', 'footers', 'quotes'),
                                          download_if_missing=True)
            return newsgroups.data[:sample_size], newsgroups.target[:sample_size], newsgroups.target_names
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                st.error("Failed to download dataset after multiple attempts.")
                return None, None, None
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            return None, None, None

def main():
    st.title("📊 Advanced Document Clustering Explorer")
    st.markdown("""
    Explore document clustering and topic modeling with interactive visualizations
    """)

    if 'results' not in st.session_state:
        st.session_state['results'] = None

    # Sidebar controls
    st.sidebar.header("Configuration")
    dataset_choice = st.sidebar.radio(
        "Data source:",
        ("Use sample dataset (20 Newsgroups)", "Upload your own documents")
    )

    if dataset_choice == "Upload your own documents":
        uploaded_files = st.sidebar.file_uploader(
            "Upload text files", 
            type=["txt", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            documents = [file.read().decode("utf-8") for file in uploaded_files]
            filenames = [file.name for file in uploaded_files]
            true_labels, target_names = None, None
    else:
        sample_size = st.sidebar.slider("Sample size", 100, 5000, 1000)
        documents, true_labels, target_names = load_sample_data(sample_size)
        if documents is not None:
            filenames = [f"doc_{i}" for i in range(len(documents))]

    # Algorithm settings
    st.sidebar.header("Algorithm Settings")
    algorithm = st.sidebar.selectbox(
        "Algorithm:",
        ("K-means Clustering", "LDA Topic Modeling", "NMF Topic Modeling")
    )
    
    n_clusters = st.sidebar.slider("Number of clusters/topics", 2, 30, 10)
    max_features = st.sidebar.slider("Max vocabulary size", 100, 5000, 1000)
    use_tsne = st.sidebar.checkbox("Use t-SNE for visualization", False)

    # Analysis execution
    if st.sidebar.button("Run Analysis") and documents:
        with st.spinner("Preprocessing..."):
            corpus = [preprocess(doc) for doc in documents]
            combined_text = ' '.join(corpus)
        
        # Feature extraction
        with st.spinner("Extracting features..."):
            if algorithm == "K-means Clustering":
                vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
            else:
                vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
            matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
        
        # Clustering
        with st.spinner("Running algorithm..."):
            if algorithm == "K-means Clustering":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = model.fit_predict(matrix)
                
                # Get top words per cluster
                top_words = {}
                for i in range(n_clusters):
                    cluster_docs = matrix[cluster_labels == i]
                    if cluster_docs.shape[0] > 0:
                        centroid = np.asarray(cluster_docs.mean(axis=0))
                        top_indices = centroid.argsort()[0, -10:][::-1]
                        top_words[i] = [str(feature_names[idx]) for idx in top_indices.flatten()]
                    else:
                        top_words[i] = []
            elif algorithm == "LDA Topic Modeling":
                model = LatentDirichletAllocation(
                    n_components=n_clusters,
                    random_state=42,
                    learning_method='online'
                )
                doc_topic_distrib = model.fit_transform(matrix)
                cluster_labels = doc_topic_distrib.argmax(axis=1)
                
                # Get top words per topic
                top_words = {}
                for topic_idx in range(n_clusters):
                    top_indices = model.components_[topic_idx].argsort()[-10:][::-1]
                    top_words[topic_idx] = [str(feature_names[i]) for i in top_indices]
            else:  # NMF
                model = NMF(n_components=n_clusters, random_state=42)
                doc_topic_distrib = model.fit_transform(matrix)
                cluster_labels = doc_topic_distrib.argmax(axis=1)
                
                # Get top words per topic
                top_words = {}
                for topic_idx in range(n_clusters):
                    top_indices = model.components_[topic_idx].argsort()[-10:][::-1]
                    top_words[topic_idx] = [str(feature_names[i]) for i in top_indices]
        
        # Visualization
        with st.spinner("Creating visualizations..."):
            if use_tsne:
                reducer = TSNE(n_components=2, random_state=42)
            else:
                reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(matrix.toarray())
            
            results_df = pd.DataFrame({
                'Document': filenames[:len(documents)],
                'Text': [doc[:200] + "..." for doc in documents],
                'Cluster': cluster_labels
            })
            
            if true_labels is not None:
                results_df['True Label'] = true_labels
            
            st.session_state['results'] = {
                'df': results_df,
                'reduced_data': reduced_data,
                'cluster_labels': cluster_labels,
                'top_words': top_words,
                'algorithm': algorithm,
                'true_labels': true_labels,
                'combined_text': combined_text,
                'model': model,
                'feature_names': feature_names,
                'target_names': target_names,
                'matrix': matrix
            }
    
    # Display results
    if st.session_state['results']:
        results = st.session_state['results']
        
        st.header("Results Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            results['df']['Cluster'].value_counts().sort_index().plot(
                kind='bar', color='skyblue', ax=ax
            )
            plt.xlabel("Cluster ID")
            plt.ylabel("Number of Documents")
            st.pyplot(fig)
            
            if results['true_labels'] is not None:
                st.subheader("Evaluation Metrics")
                ari = adjusted_rand_score(results['true_labels'], results['cluster_labels'])
                nmi = normalized_mutual_info_score(results['true_labels'], results['cluster_labels'])
                
                metrics_df = pd.DataFrame({
                    'Metric': ['Adjusted Rand Index', 'Normalized Mutual Info'],
                    'Value': [f"{ari:.4f}", f"{nmi:.4f}"]
                })
                st.table(metrics_df)
        
        with col2:
            st.subheader(f"{results['algorithm']} Visualization")
            plot_df = pd.DataFrame({
                'x': results['reduced_data'][:, 0],
                'y': results['reduced_data'][:, 1],
                'cluster': results['cluster_labels'],
                'text': results['df']['Text']
            })
            
            fig = px.scatter(
                plot_df, x='x', y='y', color='cluster',
                hover_data=['text'],
                title=f"{'t-SNE' if use_tsne else 'PCA'} Visualization"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Topic-word heatmap for topic models
        if results['algorithm'] != "K-means Clustering":
            st.header("Topic-Word Heatmap")
            plot_topic_heatmap(
                results['model'].components_,
                results['feature_names']
            )
        
        # Confusion matrix if ground truth available
        if results['true_labels'] is not None:
            st.header("Confusion Matrix")
            plot_confusion_matrix(
                results['true_labels'],
                results['cluster_labels'],
                f"{results['algorithm']} vs True Labels"
            )
        
        # Top words per cluster/topic
        st.header("Top Terms per Cluster/Topic")
        cols = st.columns(3)
        for cluster_id, words in results['top_words'].items():
            with cols[cluster_id % 3]:
                st.markdown(f"#### {results['algorithm'].split()[0]} {cluster_id}")
                st.write(", ".join(words))
                
                if results['algorithm'] != "K-means Clustering":
                    plot_wordcloud(" ".join(words), f"Topic {cluster_id}")
        
        # Corpus word cloud
        st.header("Corpus Word Cloud")
        plot_wordcloud(results['combined_text'])
        
        # Data exploration
        st.header("Data Exploration")
        st.dataframe(results['df'])
        st.markdown(get_table_download_link(results['df']), unsafe_allow_html=True)

if __name__ == "__main__":
    main()