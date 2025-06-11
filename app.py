import streamlit as st
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import threading
import time
import json

# Add the autointerp package to the Python path
sys.path.append(str(Path(__file__).parent))

from autointerp.base import Feature
from autointerp.loader import load
from autointerp.samplers import make_quantile_sampler, identity_sampler
from autointerp.automation.explainer import Explainer
from autointerp.automation.classifier import Classifier
from autointerp.automation.clients import LocalClient, OpenRouterClient
from autointerp.automation.prompts.explainer_prompt import SYSTEM, EXAMPLES
from autointerp.automation.prompts.detection_prompt import DSCORER_SYSTEM_PROMPT

# Page configuration
st.set_page_config(
    page_title="Autointerp Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "features" not in st.session_state:
    st.session_state.features = []
if "feature_dirs" not in st.session_state:
    st.session_state.feature_dirs = []
if "running_jobs" not in st.session_state:
    st.session_state.running_jobs = {}
if "results" not in st.session_state:
    st.session_state.results = {}

def load_feature_directories(base_path: str) -> List[str]:
    """Load available feature directories from the base path."""
    if not os.path.exists(base_path):
        return []
    
    directories = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            directories.append(item)
    return sorted(directories)

def create_client(client_type: str, model: str) -> Any:
    """Create the appropriate client based on type."""
    if client_type == "Local":
        return LocalClient(model)
    elif client_type == "OpenRouter":
        return OpenRouterClient(model)
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def sampler_options():
    """Create sampler selection interface."""
    st.subheader("Sampler Configuration")
    
    sampler_type = st.selectbox(
        "Sampler Type", 
        ["quantile", "identity"],
        help="Choose how to sample examples from features"
    )
    
    if sampler_type == "quantile":
        col1, col2 = st.columns(2)
        with col1:
            n_examples = st.number_input("Examples per quantile", min_value=1, value=20)
            n_quantiles = st.number_input("Number of quantiles", min_value=1, value=1)
        with col2:
            n_exclude = st.number_input("Exclude from start", min_value=0, value=0)
            n_top_exclude = st.number_input("Exclude from top", min_value=0, value=0)
        
        return make_quantile_sampler(
            n_examples=n_examples,
            n_quantiles=n_quantiles,
            n_exclude=n_exclude,
            n_top_exclude=n_top_exclude
        )
    else:
        return identity_sampler

def client_configuration():
    """Create client configuration interface."""
    st.subheader("Client Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        client_type = st.selectbox("Client Type", ["Local", "OpenRouter"])
    with col2:
        model = st.text_input("Model", value="gpt-4o-mini" if client_type == "OpenRouter" else "meta-llama/llama-3.1-8b-instruct")
    
    return client_type, model

def feature_loader_interface(base_path: str):
    """Create interface for loading features."""
    if not base_path or not os.path.exists(base_path):
        st.warning("Please provide a valid base path containing feature directories.")
        return None
    
    directories = load_feature_directories(base_path)
    if not directories:
        st.warning("No feature directories found in the specified path.")
        return None
    
    selected_dir = st.selectbox(
        "Select Feature Directory",
        directories,
        help="Choose which cached feature directory to load"
    )
    
    if selected_dir:
        full_path = os.path.join(base_path, selected_dir)
        return full_path
    
    return None

async def run_explanation_job(features: List[Feature], explainer: Explainer, job_id: str):
    """Run explanation job asynchronously."""
    try:
        st.session_state.running_jobs[job_id] = {
            "type": "explanation",
            "total": len(features),
            "completed": 0,
            "status": "running"
        }
        
        results = {}
        for i, feature in enumerate(features):
            explanation = await explainer(feature)
            results[feature.index] = explanation
            
            # Update progress
            st.session_state.running_jobs[job_id]["completed"] = i + 1
            
        st.session_state.results[job_id] = results
        st.session_state.running_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        st.session_state.running_jobs[job_id]["status"] = f"error: {str(e)}"

async def run_classification_job(features: List[Feature], classifier: Classifier, explanations: Dict[int, str], job_id: str):
    """Run classification job asynchronously."""
    try:
        st.session_state.running_jobs[job_id] = {
            "type": "classification", 
            "total": len(features),
            "completed": 0,
            "status": "running"
        }
        
        results = {}
        for i, feature in enumerate(features):
            explanation = explanations.get(feature.index, "No explanation available")
            result = await classifier(feature, "max_activating_examples", explanation)
            results[feature.index] = result
            
            # Update progress
            st.session_state.running_jobs[job_id]["completed"] = i + 1
            
        st.session_state.results[job_id] = results
        st.session_state.running_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        st.session_state.running_jobs[job_id]["status"] = f"error: {str(e)}"

def explanation_tab():
    """Create the explanation tab interface."""
    st.header("🔍 Feature Explanation")
    
    # Prompt editor
    st.subheader("Explainer Prompt")
    current_prompt = st.text_area(
        "Edit the explainer prompt (supports Python template strings):",
        value=SYSTEM,
        height=300,
        help="This prompt will be used to generate explanations for features"
    )
    
    # Sampler configuration
    sampler = sampler_options()
    
    # Client configuration  
    client_type, model = client_configuration()
    
    # Additional explainer options
    st.subheader("Explainer Options")
    col1, col2 = st.columns(2)
    with col1:
        max_or_min = st.selectbox("Activation Type", ["max", "min"])
        threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.1)
    with col2:
        insert_as_prompt = st.checkbox("Insert as prompt", value=False)
        verbose = st.checkbox("Verbose output", value=False)
    
    # Run explanation
    if st.button("🚀 Run Explanation", type="primary"):
        if not st.session_state.features:
            st.error("Please load features first.")
            return
            
        try:
            client = create_client(client_type, model)
            explainer = Explainer(
                client=client,
                max_or_min=max_or_min,
                threshold=threshold,
                insert_as_prompt=insert_as_prompt,
                verbose=verbose
            )
            
            job_id = f"explanation_{int(time.time())}"
            
            # Run in background thread
            def run_job():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_explanation_job(st.session_state.features, explainer, job_id)
                )
                loop.close()
            
            thread = threading.Thread(target=run_job)
            thread.daemon = True
            thread.start()
            
            st.success(f"Explanation job started! Job ID: {job_id}")
            
        except Exception as e:
            st.error(f"Failed to start explanation job: {str(e)}")

def classification_tab():
    """Create the classification tab interface."""
    st.header("📊 Feature Classification")
    
    # Prompt editor
    st.subheader("Classification Prompt") 
    current_prompt = st.text_area(
        "Edit the classification prompt:",
        value=DSCORER_SYSTEM_PROMPT,
        height=300,
        help="This prompt will be used to classify feature examples"
    )
    
    # Classification options
    st.subheader("Classification Options")
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch size", min_value=1, value=32)
        threshold = st.slider("Activation threshold", 0.0, 1.0, 0.3, 0.1)
    with col2:
        provide_random = st.checkbox("Provide random examples", value=True)
        provide_similar = st.checkbox("Provide similar examples", value=True)
    
    # Sampler configuration
    sampler = sampler_options()
    
    # Client configuration
    client_type, model = client_configuration()
    
    # Additional classifier options
    st.subheader("Classifier Options")
    col1, col2 = st.columns(2)
    with col1:
        n_examples_shown = st.number_input("Examples shown", min_value=1, value=10)
        method = st.selectbox("Method", ["detection"])  # Only detection supported per requirements
    with col2:
        verbose = st.checkbox("Verbose output", value=False)
    
    # Run classification
    if st.button("🚀 Run Classification", type="primary"):
        if not st.session_state.features:
            st.error("Please load features first.")
            return
            
        # Check if we have explanations
        explanation_results = None
        for job_id, results in st.session_state.results.items():
            if "explanation" in job_id:
                explanation_results = results
                break
                
        if not explanation_results:
            st.error("Please run explanations first before classification.")
            return
            
        try:
            client = create_client(client_type, model)
            classifier = Classifier(
                client=client,
                n_examples_shown=n_examples_shown,
                method=method,
                threshold=threshold,
                verbose=verbose
            )
            
            job_id = f"classification_{int(time.time())}"
            
            # Run in background thread
            def run_job():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    run_classification_job(st.session_state.features, classifier, explanation_results, job_id)
                )
                loop.close()
            
            thread = threading.Thread(target=run_job)
            thread.daemon = True
            thread.start()
            
            st.success(f"Classification job started! Job ID: {job_id}")
            
        except Exception as e:
            st.error(f"Failed to start classification job: {str(e)}")

def job_progress_sidebar():
    """Display running jobs in the right sidebar."""
    with st.sidebar:
        st.header("🏃 Running Jobs")
        
        if not st.session_state.running_jobs:
            st.info("No running jobs")
            return
            
        for job_id, job_info in st.session_state.running_jobs.items():
            if job_info["status"] == "completed":
                st.success(f"✅ {job_id[:20]}...")
                continue
            elif job_info["status"].startswith("error"):
                st.error(f"❌ {job_id[:20]}...")
                st.text(job_info["status"])
                continue
                
            # Show progress for running jobs
            progress = job_info["completed"] / job_info["total"]
            st.text(f"🔄 {job_info['type'].title()}")
            st.progress(progress)
            st.text(f"{job_info['completed']}/{job_info['total']} features")

def main():
    """Main application."""
    st.title("🧠 Autointerp Dashboard")
    
    # Sidebar for navigation and job progress
    with st.sidebar:
        st.header("📁 Feature Loading")
        
        # Get base path from command line or user input
        if len(sys.argv) > 1:
            base_path = sys.argv[1]
            st.text(f"Base path: {base_path}")
        else:
            base_path = st.text_input("Base path to feature directories:", placeholder="/path/to/features")
        
        # Feature loader
        if base_path:
            selected_path = feature_loader_interface(base_path)
            
            if selected_path and st.button("Load Features"):
                with st.spinner("Loading features..."):
                    try:
                        sampler = make_quantile_sampler(n_examples=20, n_quantiles=1)
                        features = load(
                            selected_path,
                            sampler=sampler,
                            ctx_len=64,
                            max_examples=100,
                            load_similar_non_activating=10,
                            load_random_non_activating=10
                        )
                        st.session_state.features = features
                        st.success(f"Loaded {len(features)} features!")
                    except Exception as e:
                        st.error(f"Failed to load features: {str(e)}")
        
        st.divider()
        
        # Tab selection
        st.header("📋 Navigation")
        tab_selection = st.radio("Select Tab:", ["Explain", "Classify"])
    
    # Job progress in right column
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main content area
        if tab_selection == "Explain":
            explanation_tab()
        else:
            classification_tab()
    
    with col2:
        job_progress_sidebar()
    
    # Display results
    if st.session_state.results:
        st.header("📈 Results")
        
        for job_id, results in st.session_state.results.items():
            with st.expander(f"Results: {job_id}"):
                if isinstance(results, dict):
                    st.json(results)
                else:
                    st.text(str(results))

if __name__ == "__main__":
    main()