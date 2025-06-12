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

def feature_loading_component(base_path: str, key_prefix: str):
    """Create feature loading interface with sampler configuration."""
    st.subheader("🔄 Feature Loading")
    
    if not base_path or not os.path.exists(base_path):
        st.warning("Please provide a valid base path containing feature directories in the sidebar.")
        return None, None
    
    # Caching job directory selection
    caching_jobs = load_feature_directories(base_path)
    if not caching_jobs:
        st.warning("No caching job directories found in the specified path.")
        return None, None
    
    selected_caching_job = st.selectbox(
        "Select Caching Job",
        caching_jobs,
        help="Choose which caching job directory to load from",
        key=f"{key_prefix}_caching_job"
    )
    
    full_path = None
    if selected_caching_job:
        caching_job_path = os.path.join(base_path, selected_caching_job)
        
        # Model layer directory selection
        model_layers = load_feature_directories(caching_job_path)
        if model_layers:
            # Add option to select all layers
            layer_options = ["All Layers"] + model_layers
            selected_layer = st.selectbox(
                "Select Model Layer",
                layer_options,
                help="Choose specific model layer or 'All Layers' to run on all layers",
                key=f"{key_prefix}_model_layer"
            )
            
            if selected_layer == "All Layers":
                full_path = caching_job_path
                st.info(f"📁 Will load from all layers in: {selected_caching_job}")
            else:
                full_path = os.path.join(caching_job_path, selected_layer)
                st.info(f"📁 Will load from: {selected_caching_job}/{selected_layer}")
        else:
            # No subdirectories found, use the caching job directory directly
            full_path = caching_job_path
            st.info(f"📁 No model layer subdirectories found. Using: {selected_caching_job}")
    else:
        full_path = None

    # Sampler configuration
    st.subheader("📊 Sampler Configuration")
    st.text("Quantile Sampler")
    
    col1, col2 = st.columns(2)
    with col1:
        n_examples = st.number_input("Examples per quantile", min_value=1, value=20, key=f"{key_prefix}_n_examples")
        n_quantiles = st.number_input("Number of quantiles", min_value=1, value=1, key=f"{key_prefix}_n_quantiles")
    with col2:
        n_exclude = st.number_input("Exclude from start", min_value=0, value=0, key=f"{key_prefix}_n_exclude")
        n_top_exclude = st.number_input("Exclude from top of each quantile", min_value=0, value=0, key=f"{key_prefix}_n_top_exclude")
    
    sampler = make_quantile_sampler(
        n_examples=n_examples,
        n_quantiles=n_quantiles,
        n_exclude=n_exclude,
        n_top_exclude=n_top_exclude
    )
    
    # Additional loading options
    st.subheader("⚙️ Loading Options")
    col1, col2 = st.columns(2)
    with col1:
        ctx_len = st.number_input("Example context length", min_value=16, value=64, key=f"{key_prefix}_ctx_len")
        max_examples = st.number_input("Max examples to load per feature", min_value=1, value=1000, key=f"{key_prefix}_max_examples")
    with col2:
        load_similar = st.number_input("Similar non-activating", min_value=0, value=0, key=f"{key_prefix}_similar")
        load_random = st.number_input("Random non-activating", min_value=0, value=0, key=f"{key_prefix}_random")
    
    # Load features button
    if full_path and st.button(f"🚀 Load Features", key=f"{key_prefix}_load", type="primary"):
        with st.spinner("Loading features..."):
            try:
                features = load(
                    full_path,
                    sampler=sampler,
                    ctx_len=ctx_len,
                    max_examples=max_examples,
                    load_similar_non_activating=load_similar,
                    load_random_non_activating=load_random,
                    pbar="streamlit"
                )
                st.success(f"✅ Loaded {len(features)} features!")
                return features, sampler
            except Exception as e:
                st.error(f"❌ Failed to load features: {str(e)}")
                return None, None
    
    return None, None

def client_configuration():
    """Create client configuration interface."""
    st.subheader("Client Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        client_type = st.selectbox("Client Type", ["OpenRouter", "Local"])
    with col2:
        model = st.text_input("Model", value="meta-llama/Llama-3.3-70B-Instruct" if client_type == "OpenRouter" else "meta-llama/llama-3.1-8b-instruct")
    
    return client_type, model


async def run_explanation_job(features: List[Feature], explainer: Explainer, job_id: str, save_dir: str = None):
    """Run explanation job asynchronously using gather for speed."""

    try:
        st.session_state.running_jobs[job_id] = {
            "type": "explanation",
            "total": len(features),
            "completed": 0,
            "status": "running"
        }
        
        # Use asyncio.gather for parallel processing
        async def process_feature(feature):
            explanation = await explainer(feature)
            # Update progress atomically
            st.session_state.running_jobs[job_id]["completed"] += 1
            return feature.index, explanation
        
        # Run all features in parallel
        results_list = await asyncio.gather(*[process_feature(feature) for feature in features])
        results = dict(results_list)
        
        st.session_state.results[job_id] = results
        st.session_state.running_jobs[job_id]["status"] = "completed"
        
        # Save results to directory if specified
        if save_dir and os.path.exists(save_dir):
            save_path = os.path.join(save_dir, f"{job_id}_explanations.json")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            st.session_state.running_jobs[job_id]["save_path"] = save_path
        
    except Exception as e:
        st.session_state.running_jobs[job_id]["status"] = f"error: {str(e)}"

async def run_classification_job(features: List[Feature], classifier: Classifier, explanations: Dict[int, str], job_id: str, save_dir: str = None):
    """Run classification job asynchronously using gather for speed."""
    try:
        st.session_state.running_jobs[job_id] = {
            "type": "classification", 
            "total": len(features),
            "completed": 0,
            "status": "running"
        }
        
        # Use asyncio.gather for parallel processing
        async def process_feature(feature):
            explanation = explanations.get(feature.index, "No explanation available")
            result = await classifier(feature, "max_activating_examples", explanation)
            # Update progress atomically
            st.session_state.running_jobs[job_id]["completed"] += 1
            return feature.index, result
        
        # Run all features in parallel
        results_list = await asyncio.gather(*[process_feature(feature) for feature in features])
        results = dict(results_list)
            
        st.session_state.results[job_id] = results
        st.session_state.running_jobs[job_id]["status"] = "completed"
        
        # Save results to directory if specified
        if save_dir and os.path.exists(save_dir):
            save_path = os.path.join(save_dir, f"{job_id}_classifications.json")
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            st.session_state.running_jobs[job_id]["save_path"] = save_path
        
    except Exception as e:
        st.session_state.running_jobs[job_id]["status"] = f"error: {str(e)}"

def explanation_tab(base_path: str, save_dir: str = None):
    """Create the explanation tab interface."""
    st.header("🔍 Feature Explanation")
    
    # Feature loading
    features, sampler = feature_loading_component(base_path, "explain")
    
    st.divider()
    
    # Prompt editor
    st.subheader("💬 Explain")

    with st.expander("Edit explanation prompt"):
    
        # Show full conversation structure for editing
        from autointerp.automation.prompts.explainer_prompt import build_prompt
        example_prompt = build_prompt("Example 1: Sample <<text>> with markers")
        updated_prompt = prompt_editor(example_prompt, "Explainer_Prompt")
        
    # Client configuration  
    client_type, model = client_configuration()
    
    col1, col2 = st.columns(2)
    with col1:
        max_or_min = st.selectbox("Activation Type", ["max", "min"])
        threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.1)
    with col2:
        insert_as_prompt = st.checkbox("Insert as prompt", value=False)
        verbose = st.checkbox("Verbose output", value=False)
    
    # Run explanation
    if st.button("🚀 Run Explanation", key="run_explanation", type="primary"):
        if not features:
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
                    run_explanation_job(features, explainer, job_id, save_dir)
                )
                loop.close()
            
            thread = threading.Thread(target=run_job)
            thread.daemon = True
            thread.start()
            
            st.success(f"Explanation job started! Job ID: {job_id}")
            
        except Exception as e:
            st.error(f"Failed to start explanation job: {str(e)}")

def classification_tab(base_path: str, save_dir: str = None):
    """Create the classification tab interface."""
    st.header("📊 Feature Classification")
    
    # Feature loading
    features, sampler = feature_loading_component(base_path, "classify")
    
    st.divider()
    
    # Prompt editor
    st.subheader("💬 Classification") 
    
    with st.expander("Edit classification prompt"):
        # Show full conversation structure for editing
        from autointerp.automation.prompts.detection_prompt import prompt
        example_prompt = prompt("Example 1: Sample text", "Sample explanation")
        updated_prompt = prompt_editor(example_prompt, "Classification_Prompt")
    
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch size", min_value=1, value=32)
        threshold = st.slider("Activation threshold", 0.0, 1.0, 0.3, 0.1)
    with col2:
        provide_random = st.checkbox("Provide random examples", value=True)
        provide_similar = st.checkbox("Provide similar examples", value=True)
    
    # Client configuration
    client_type, model = client_configuration()
    
    # Additional classifier options
    st.subheader("⚙️ Classifier Options")
    col1, col2 = st.columns(2)
    with col1:
        n_examples_shown = st.number_input("Examples shown", min_value=1, value=10)
        method = st.selectbox("Method", ["detection"])  # Only detection supported per requirements
    with col2:
        verbose = st.checkbox("Verbose output", value=False)
    
    # Run classification
    if st.button("🚀 Run Classification", key="run_classification", type="primary"):
        if not features:
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
                    run_classification_job(features, classifier, explanation_results, job_id, save_dir)
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
                if "save_path" in job_info:
                    st.text(f"💾 Saved to: {os.path.basename(job_info['save_path'])}")
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

def messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Convert messages to a single editable text format."""
    text_parts = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        text_parts.append(f"[[{role}]]\n{content}")
    return "\n\n".join(text_parts)

def text_to_messages(text: str) -> List[Dict[str, str]]:
    """Parse text format back to messages."""
    messages = []
    parts = text.split("[[")
    
    for part in parts[1:]:  # Skip first empty part
        if "]]" not in part:
            continue
        
        role_end = part.find("]]")
        role = part[:role_end].strip().lower()
        content = part[role_end + 2:].strip()
        
        if role and content:
            messages.append({"role": role, "content": content})
    
    return messages

def check_messages_valid(messages: List[Dict[str, str]]) -> bool:
    """Check if messages are valid."""
    if not messages:
        return False
        
    # Must start with system message
    if messages[0]["role"] != "system":
        return False
        
    # Check alternating pattern and valid roles
    for i, msg in enumerate(messages[1:], 1):
        if msg["role"] not in ["user", "assistant"]:
            return False
            
        # Check alternating pattern
        if i > 0 and msg["role"] == messages[i-1]["role"]:
            return False
            
    return True

def prompt_editor(messages: List[Dict[str, str]], title: str):
    """Main prompt editing interface with toggle between edit and preview modes."""
    # Initialize session state for this conversation
    edit_key = f"{title}_edit_mode"
    text_key = f"{title}_text"
    
    if edit_key not in st.session_state:
        st.session_state[edit_key] = True
    
    if text_key not in st.session_state:
        st.session_state[text_key] = messages_to_text(messages)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("📝 Edit Mode", key=f"{title}_edit_btn", type="primary" if st.session_state[edit_key] else "secondary"):
            st.session_state[edit_key] = True
            
    with col2:
        if st.button("👁️ Preview Mode", key=f"{title}_preview_btn", type="primary" if not st.session_state[edit_key] else "secondary"):
            st.session_state[edit_key] = False
    
    if st.session_state[edit_key]:
        # Edit mode - show text area
        st.markdown("**Edit the conversation:**")
        st.markdown("Use `[[SYSTEM]]`, `[[USER]]`, `[[ASSISTANT]]` to mark different parts")
        
        edited_text = st.text_area(
            "Conversation:",
            value=st.session_state[text_key],
            height=400,
            key=f"{title}_textarea",
            help="Edit the conversation using role markers"
        )
        
        # Update session state when text changes
        st.session_state[text_key] = edited_text
        
        # Try to parse and return updated messages
        try:
            parsed_messages = text_to_messages(edited_text)
            if parsed_messages:
                return parsed_messages
        except:
            pass
    
    else:
        # Preview mode - auto-parse and show formatted output
        st.markdown("**Formatted Conversation Preview:**")
        
        try:
            parsed_messages = text_to_messages(st.session_state[text_key])
            
            if parsed_messages:
                # Display parsed count
                if check_messages_valid(parsed_messages):
                    st.info(f"📊 Parsed {len(parsed_messages)} messages")
                else:
                    st.error("⚠️ Invalid messages found. Check your formatting.")
                
                # Display as JSON with proper wrapping
                st.markdown("**JSON Format:**")
                json_str = json.dumps(parsed_messages, indent=2, ensure_ascii=False)
                st.code(json_str, language="json", wrap_lines=True)
                
                return parsed_messages
            else:
                st.warning("⚠️ No valid messages found. Check your formatting.")
                
        except Exception as e:
            st.error(f"❌ Error parsing conversation: {str(e)}")
            st.code(st.session_state[text_key], language="text", wrap_lines=True)
    
    return messages  # Return original if no updates

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
        
        # Save directory input
        save_dir = st.text_input("Save directory for results:", placeholder="/path/to/save/results", help="Directory to save job results as JSON files")
        
        st.divider()
        
        # Tab selection
        st.header("📋 Navigation")
        tab_selection = st.radio("Select Tab:", ["Explain", "Classify"])
    
    # Job progress in right column
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main content area
        if tab_selection == "Explain":
            explanation_tab(base_path if base_path else "", save_dir if save_dir else None)
        else:
            classification_tab(base_path if base_path else "", save_dir if save_dir else None)
    
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