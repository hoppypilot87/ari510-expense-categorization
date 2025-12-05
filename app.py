import gradio as gr
import pandas as pd

import model_predict as mp  # our helper module


# ---------------------------------------------------------
# Single prediction callback
# ---------------------------------------------------------

def single_predict(vendor, description, amount, payment_method, city, state):
    """
    Wrapper for Gradio: returns both the best label and per-class probabilities.
    """
    # Gradio Number can pass None if empty; guard against that
    if amount is None:
        amount = 0.0

    best_label, proba_dict = mp.predict_category_with_proba(
        vendor=vendor,
        description=description,
        amount=float(amount),
        payment_method=payment_method,
        city=city,
        state=state,
    )

    # First output: plain label string
    # Second output: dict -> Gradio Label shows probability bars
    return best_label, proba_dict


# ---------------------------------------------------------
# Batch prediction callback
# ---------------------------------------------------------

def batch_predict(csv_path):
    """
    Batch predict from a CSV uploaded file path.

    Expects columns:
    vendor, description, amount, payment_method, city, state
    """
    if csv_path is None:
        return "Please upload a CSV file.", None

    df = pd.read_csv(csv_path)

    required_cols = [
        "vendor",
        "description",
        "amount",
        "payment_method",
        "city",
        "state",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return (
            f"Missing required columns in CSV: {', '.join(missing)}",
            None,
        )

    model = mp.load_model()

    preds = model.predict(df[required_cols])
    labels = [mp.LABEL_CLASSES[int(i)] for i in preds]

    out_df = df.copy()
    out_df["predicted_category"] = labels

    return "Batch prediction complete.", out_df


# ---------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        """
# 💳 Expense Category Predictor

Use this tool to classify expenses into spending categories using
**vendor**, **description**, **amount**, and supporting metadata.

- 🧪 **Single Prediction**: Try one transaction at a time and see confidence bars.
- 📑 **Batch Prediction**: Upload a CSV of many transactions to label them all at once.
        """
    )

    with gr.Tab("Single Prediction"):
        gr.Markdown(
            "### Predict a single transaction\n"
            "Enter the transaction details on the left. "
            "The model will predict the most likely spending category and show its confidence."
        )

        with gr.Row():
            # Inputs column
            with gr.Column(scale=3):
                vendor_in = gr.Textbox(
                    label="Vendor",
                    placeholder="e.g., JetBlue, Shell Gas Station, Starbucks",
                )
                desc_in = gr.Textbox(
                    label="Description",
                    placeholder="e.g., airline ticket, coffee, gas purchase",
                )
                amount_in = gr.Number(
                    label="Amount",
                    precision=2,
                    value=0.0,
                )
                pay_in = gr.Dropdown(
                    label="Payment Method",
                    choices=["Credit Card", "Debit Card", "Cash", "Digital Wallet"],
                    value="Credit Card",
                )
                city_in = gr.Textbox(label="City", placeholder="e.g., Detroit")
                state_in = gr.Textbox(label="State", placeholder="e.g., MI")

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    submit_btn = gr.Button("Submit", variant="primary")

            # Outputs column
            with gr.Column(scale=2):
                pred_text = gr.Textbox(
                    label="Predicted Category",
                    interactive=False,
                    placeholder="Prediction will appear here...",
                )
                prob_label = gr.Label(
                    label="Confidence Across Categories",
                    visible=True,
                )
                flag_btn = gr.Button("Flag", variant="secondary")

        # Wire buttons
        submit_btn.click(
            single_predict,
            inputs=[vendor_in, desc_in, amount_in, pay_in, city_in, state_in],
            outputs=[pred_text, prob_label],
        )

        def clear_inputs():
            return "", "", 0.0, "Credit Card", "", ""

        clear_btn.click(
            clear_inputs,
            outputs=[vendor_in, desc_in, amount_in, pay_in, city_in, state_in],
        )

        # For now, Flag just prints to console (could later log to a file)
        def flag_example(vendor, description, amount, payment_method, city, state, pred):
            print("FLAGGED EXAMPLE:")
            print(
                {
                    "vendor": vendor,
                    "description": description,
                    "amount": amount,
                    "payment_method": payment_method,
                    "city": city,
                    "state": state,
                    "predicted_category": pred,
                }
            )
            return gr.update()

        flag_btn.click(
            flag_example,
            inputs=[
                vendor_in,
                desc_in,
                amount_in,
                pay_in,
                city_in,
                state_in,
                pred_text,
            ],
            outputs=[],
        )

    # ------------------ Batch Tab ------------------
    with gr.Tab("Batch Prediction"):
        gr.Markdown(
            "### Batch predict from CSV\n"
            "Upload a CSV with columns: "
            "`vendor, description, amount, payment_method, city, state`."
        )

        with gr.Row():
            with gr.Column(scale=2):
                file_in = gr.File(
                    label="Upload CSV",
                    type="filepath",  # Gradio v4+ uses 'filepath' or 'binary'
                )
                batch_btn = gr.Button("Run Batch Prediction", variant="primary")
            with gr.Column(scale=3):
                status_out = gr.Textbox(
                    label="Status",
                    interactive=False,
                    placeholder="Status messages will appear here...",
                )
                table_out = gr.Dataframe(
                    label="Predicted Transactions",
                    interactive=False,
                )

        batch_btn.click(
            batch_predict,
            inputs=file_in,
            outputs=[status_out, table_out],
        )

if __name__ == "__main__":
    demo.launch()