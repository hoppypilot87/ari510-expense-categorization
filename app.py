import gradio as gr
import numpy as np
from typing import List, Optional
import pandas as pd

from model_predict import (
    predict_category_from_vector,
    demo_random_prediction,
    load_artifacts,
    LABEL_CLASSES,
)


def predict_from_csv_vector(csv_numbers: str) -> str:
    """Accept a comma-separated list of numeric feature values and return a category with styling."""
    if not csv_numbers.strip():
        return "⚠️ Please enter feature values"
    
    try:
        # Parse numbers from the input string
        parts = [p.strip() for p in csv_numbers.split(",") if p.strip() != ""]
        x = np.array([float(p) for p in parts])
    except Exception as e:
        return f"❌ Error parsing input: {e}"

    try:
        result = predict_category_from_vector(x)
        return f"✓ {result}"
    except Exception as e:
        return f"❌ Prediction error: {e}"


def run_demo_random(n: int = 3) -> str:
    try:
        demo_random_prediction(n_samples=n)
        return "✓ Demo completed — check server logs for detailed output"
    except Exception as e:
        return f"❌ Demo error: {e}"


def load_feature_names() -> Optional[List[str]]:
    """Return the scaler's feature names if available, else None."""
    try:
        _, scaler = load_artifacts()
        return list(getattr(scaler, "feature_names_in_", [])) or None
    except Exception:
        return None


def build_example_vector(feature_names: Optional[List[str]]) -> str:
    """Create a simple example vector (zeros) matching feature count."""
    if not feature_names:
        return ""
    return ", ".join(["0" for _ in feature_names])


def predict_batch_csv(csv_file) -> pd.DataFrame:
    """Process a CSV file and return predictions for each row."""
    if csv_file is None:
        return pd.DataFrame({"error": ["Please upload a CSV file"]})
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if 'amount' column exists
        if 'amount' not in df.columns:
            return pd.DataFrame({
                "error": [f"CSV must have an 'amount' column. Found columns: {', '.join(df.columns)}"]
            })
        
        # Run predictions on each row
        predictions = []
        for idx, row in df.iterrows():
            try:
                amount = float(row['amount'])
                pred = predict_category_from_vector(np.array([amount]))
                predictions.append({
                    'Row #': idx + 1,
                    'Amount': f"${amount:.2f}",
                    'Predicted Category': pred
                })
            except Exception as e:
                predictions.append({
                    'Row #': idx + 1,
                    'Amount': str(row['amount']),
                    'Predicted Category': f"❌ Error: {str(e)}"
                })
        
        return pd.DataFrame(predictions)
    
    except Exception as e:
        return pd.DataFrame({
            "error": [f"Error processing file: {str(e)}"]
        })


feature_names = load_feature_names()
example_vector = build_example_vector(feature_names)

# Custom CSS for better visual design
custom_css = """
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

#title {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

#title h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

#title p {
    margin: 0;
    font-size: 1.1rem;
    opacity: 0.95;
}

.instruction-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #667eea;
    color: #1e293b;
}

.instruction-card h3, .instruction-card ol, .instruction-card li, .instruction-card strong {
    color: #1e293b !important;
}

.info-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    color: #1e293b;
}

.info-box h3, .info-box p, .info-box strong {
    color: #1e293b !important;
}

.class-badge {
    display: inline-block;
    background: #667eea;
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 16px;
    margin: 0.25rem;
    font-size: 0.9rem;
}

textarea, input {
    border: 2px solid #e2e8f0 !important;
    border-radius: 8px !important;
    transition: border-color 0.2s !important;
}

textarea:focus, input:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}
"""

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    
    # Header
    with gr.Row(elem_id="title"):
        gr.Markdown(
            """
            # 💰 Expense Category Predictor
            Intelligent classification of expense categories using machine learning
            """
        )
    
    # Main content area
    with gr.Row():
        # Left column - Input and prediction
        with gr.Column(scale=3):
            # Quick start guide
            gr.Markdown(
                """
                <div class="instruction-card">
                <h3 style="margin-top: 0; color: #1e293b;">🚀 Quick Start</h3>
                <ol style="margin-bottom: 0; line-height: 1.8;">
                <li><strong>Have data?</strong> Paste your comma-separated feature values below</li>
                <li><strong>Get prediction!</strong> Click the "Predict Category" button</li>
                </ol>
                </div>
                """
            )
            
            # Input area
            inp = gr.Textbox(
                label="📊 Feature Vector Input",
                placeholder="Enter value ",
                lines=4,
                show_label=True,
            )
            
            # Action buttons
            with gr.Row():
                predict_btn = gr.Button("🎯 Predict Category", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
            
            # Output area
            out = gr.Textbox(
                label="🎉 Prediction Result",
                interactive=False,
                lines=2,
                show_label=True,
            )
            
            # Collapsible help section
            with gr.Accordion("💡 Need Help?", open=False):
                gr.Markdown(
                    """
                    ### Tips for Best Results
                    
                    - **Feature Order Matters**: Ensure your values match the expected feature order shown in the sidebar
                    - **Format**: Use commas to separate values (e.g., `1.5, 2.0, 3.5`)
                    - **No Trailing Commas**: Avoid extra commas at the end
                    - **Real Data**: For authentic examples, use the helper script in the repository
                    
                    ### Troubleshooting
                    
                    - **Parsing Error**: Check for non-numeric characters or formatting issues
                    - **Wrong Prediction**: Verify you have the correct number of features
                    - **Need Feature Count**: Check the sidebar for the expected number of features
                    """
                )
        
        # Right column - Information sidebar
        with gr.Column(scale=2):
            gr.Markdown(
                """
                <div class="info-box">
                <h3 style="margin-top: 0; color: #1e293b;">📋 Model Information</h3>
                </div>
                """
            )
            
            # Predictable classes
            gr.Markdown("### 🏷️ Predictable Categories")
            classes_html = "".join([f'<span class="class-badge">{c}</span>' for c in LABEL_CLASSES])
            gr.HTML(f'<div style="padding: 1rem 0;">{classes_html}</div>')
            
            gr.Markdown("---")
            
            # Feature information
            if feature_names:
                gr.Markdown(f"### 🔢 Expected Features ({len(feature_names)} total)")
                gr.Textbox(
                    value="\n".join([f"{i+1:2d}. {name}" for i, name in enumerate(feature_names)]),
                    label="Feature Order",
                    lines=min(15, len(feature_names)),
                    interactive=False,
                    max_lines=20,
                )
            else:
                gr.Markdown(
                    """
                    <div style="background: #fef3c7; padding: 1rem; border-radius: 6px; border-left: 4px solid #f59e0b;">
                    ⚠️ <strong>Feature names unavailable</strong><br/>
                    Ensure <code>models/category_scaler.pkl</code> exists
                    </div>
                    """
                )
    
    # Wire up interactions
    predict_btn.click(fn=predict_from_csv_vector, inputs=inp, outputs=out)
    clear_btn.click(fn=lambda: "", inputs=None, outputs=inp)
    
    # CSV Batch Prediction Section
    gr.Markdown("---")
    gr.Markdown(
        """
        <div class="info-box">
        <h3 style="margin-top: 0;">📁 Batch Predictions (CSV Upload)</h3>
        <p style="margin-bottom: 0;">Upload a CSV file with an 'amount' column to get predictions for multiple rows at once</p>
        </div>
        """
    )
    
    with gr.Row():
        csv_file = gr.File(label="📤 Upload CSV File", file_types=[".csv"])
        predict_csv_btn = gr.Button("🚀 Run Batch Prediction", variant="primary")
    
    csv_results = gr.Dataframe(label="📊 Prediction Results", interactive=False)
    
    # Wire up CSV batch prediction
    predict_csv_btn.click(fn=predict_batch_csv, inputs=csv_file, outputs=csv_results)

if __name__ == "__main__":
    demo.launch()