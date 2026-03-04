import scikit-learn as sklearn
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import base64
import io
import os
from PIL import Image

st.set_page_config(
    page_title="ShopSimilar - Product Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

LABEL_MAP = {
    0: "T-shirt/Top", 1: "Trouser",  2: "Pullover", 3: "Dress",
    4: "Coat",        5: "Sandal",   6: "Shirt",    7: "Sneaker",
    8: "Bag",         9: "Ankle Boot"
}
CONFIDENCE_THRESHOLD = 0.60

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .product-card {
        background: white; border-radius: 12px; padding: 12px;
        margin: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .conf-high     { background:#e6f4ea; color:#2d6a4f; padding:2px 8px; border-radius:12px; font-size:0.75em; }
    .conf-low      { background:#fff3cd; color:#856404; padding:2px 8px; border-radius:12px; font-size:0.75em; }
    .conf-verified { background:#e3f2fd; color:#1565c0; padding:2px 8px; border-radius:12px; font-size:0.75em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "knn_model.pkl"), "rb") as f:
        knn = pickle.load(f)
    with open(os.path.join(base_dir, "pca_model.pkl"), "rb") as f:
        pca = pickle.load(f)
    return knn, pca

@st.cache_data(show_spinner="Loading catalogue...")
def load_catalogue():
    base_dir     = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "catalogue.json"), "r") as f:
        catalogue = json.load(f)
    pca_features = np.load(os.path.join(base_dir, "catalogue_pca_features.npy"))
    labels       = np.load(os.path.join(base_dir, "catalogue_labels.npy"))
    return catalogue, pca_features, labels

try:
    knn_model, pca_model           = load_model()
    catalogue, cat_pca, cat_labels = load_catalogue()
    model_loaded = True
except FileNotFoundError as e:
    model_loaded  = False
    missing_file  = str(e)

def b64_to_pil(b64_str):
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))

def get_recommendations(query_idx, k=6):
    from sklearn.metrics.pairwise import euclidean_distances
    dists              = euclidean_distances(cat_pca[query_idx].reshape(1, -1), cat_pca)[0]
    dists[query_idx]   = np.inf
    top_indices        = np.argsort(dists)[:k]
    return [(i, catalogue[i], float(dists[i])) for i in top_indices]

def conf_badge(item):
    if item["source"] == "labelled":
        return "✅ Verified"
    elif item["confidence"] >= CONFIDENCE_THRESHOLD:
        return f"🔵 {item['confidence']*100:.0f}% confident"
    else:
        return "⚠️ Uncertain"

if not model_loaded:
    st.error(f"Model files not found: {missing_file}")
    st.markdown("Make sure all .pkl, .json and .npy files are in the same folder as app.py")
    st.stop()

with st.sidebar:
    st.markdown("## 🛍️ ShopSimilar")
    st.markdown("*AI-powered product recommendations*")
    st.markdown("---")
    all_categories    = ["All Categories"] + list(LABEL_MAP.values())
    selected_category = st.selectbox("Browse by category", all_categories)
    show_uncertain    = st.checkbox("Show uncertain items", value=True)
    n_recs            = st.select_slider("Recommendations to show", options=[3, 4, 6, 8, 12], value=6)
    st.markdown("---")
    st.markdown(f"""
### Model Info
| Setting | Value |
|---------|-------|
| Algorithm | k-NN |
| k | `{knn_model.n_neighbors}` |
| Metric | `{knn_model.metric.capitalize()}` |
| PCA components | `{pca_model.n_components_}` |
| Catalogue size | `{len(catalogue)}` items |
    """)

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = None

def show_product_page(sel_idx):
    sel_item = catalogue[sel_idx]
    if st.button("Back to Catalogue"):
        st.session_state.selected_idx = None
        st.rerun()
    st.markdown("---")
    st.markdown("## Viewing Product")
    col_img, col_info = st.columns([1, 3], gap="large")
    with col_img:
        st.image(b64_to_pil(sel_item["image_b64"]), width=200)
    with col_info:
        st.markdown(f"### {sel_item['category']}")
        st.markdown(f"**Status:** {conf_badge(sel_item)}")
        st.markdown(f"**Item ID:** `{sel_item['id']}`")
        st.markdown(f"**Source:** {'Labelled (verified)' if sel_item['source'] == 'labelled' else 'Auto-classified by k-NN'}")
        if sel_item["source"] == "predicted":
            if sel_item["confidence"] < CONFIDENCE_THRESHOLD:
                st.warning("Category uncertain - recommendations based on visual similarity only.")
            else:
                st.info("Category predicted by k-NN. Recommendations reflect visual similarity.")
    st.markdown("---")
    st.markdown(f"## Similar Items - Top {n_recs} Recommendations")
    st.caption(f"Ranked by Euclidean distance in PCA-{pca_model.n_components_} feature space")
    recs     = get_recommendations(sel_idx, k=n_recs)
    rec_cols = st.columns(n_recs)
    for col, (rec_cat_idx, rec_item, dist) in zip(rec_cols, recs):
        with col:
            st.image(b64_to_pil(rec_item["image_b64"]), use_container_width=True)
            if rec_item["source"] == "predicted" and rec_item["confidence"] < CONFIDENCE_THRESHOLD:
                st.caption("Category unclear")
            else:
                st.caption(f"**{rec_item['category']}**")
            st.caption(f"Similarity: `{1/(1+dist):.3f}`")
            st.caption(conf_badge(rec_item))
            if st.button("View", key=f"rec_{rec_cat_idx}_{dist:.3f}"):
                st.session_state.selected_idx = rec_cat_idx
                st.rerun()
    st.markdown("---")
    if st.button("Back to Catalogue "):
        st.session_state.selected_idx = None
        st.rerun()

def show_browse_page():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:30px;
    border-radius:16px;margin-bottom:24px;text-align:center;">
    <h1 style="color:white;margin:0;">🛍️ ShopSimilar</h1>
    <p style="color:rgba(255,255,255,0.85);font-size:1.1em;margin:8px 0 0 0;">
    Discover visually similar products powered by AI</p></div>
    """, unsafe_allow_html=True)

    total           = len(catalogue)
    labelled_count  = sum(1 for i in catalogue if i["source"] == "labelled")
    predicted_count = total - labelled_count
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Products",  f"{total:,}")
    c2.metric("Verified Items",  f"{labelled_count:,}")
    c3.metric("Auto-classified", f"{predicted_count:,}")
    c4.metric("Categories",      "10")
    st.markdown("---")

    filtered_items = []
    for i, item in enumerate(catalogue):
        if selected_category != "All Categories" and item["category"] != selected_category:
            continue
        if not show_uncertain and item["source"] == "predicted" and item["confidence"] < CONFIDENCE_THRESHOLD:
            continue
        filtered_items.append((i, item))

    st.markdown(f"### Product Catalogue - *{len(filtered_items)} items*")

    if not filtered_items:
        st.info("No products match the current filters.")
        return

    COLS         = 5
    display_items = filtered_items[:50]
    for row_start in range(0, len(display_items), COLS):
        row_items = display_items[row_start:row_start + COLS]
        cols      = st.columns(COLS, gap="small")
        for col, (cat_idx, item) in zip(cols, row_items):
            with col:
                st.image(b64_to_pil(item["image_b64"]), use_container_width=True)
                if item["source"] == "predicted" and item["confidence"] < CONFIDENCE_THRESHOLD:
                    st.caption("Category unclear")
                else:
                    st.caption(f"**{item['category']}**")
                st.caption(conf_badge(item))
                if st.button("View Similar", key=f"browse_{cat_idx}"):
                    st.session_state.selected_idx = cat_idx
                    st.rerun()

    if len(filtered_items) > 50:
        st.info(f"Showing 50 of {len(filtered_items)} items. Use the category filter to narrow results.")

if st.session_state.selected_idx is not None:
    show_product_page(st.session_state.selected_idx)
else:
    show_browse_page()
