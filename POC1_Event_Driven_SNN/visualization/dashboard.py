from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="POC1 Event-Driven AI", layout="wide")
st.title("POC1 — Event-Driven AI")
p = Path("results/latest/sparsity_results.csv")
if not p.exists():
    st.warning("Run training and the sparsity sweep first.")
    st.stop()
df = pd.read_csv(p)
st.subheader("Compute reduction proxy")
st.line_chart(df.set_index("event_density")[["compute_reduction_proxy"]])
st.subheader("Accuracy")
st.line_chart(df.set_index("event_density")[["ann_accuracy", "snn_accuracy"]])
st.subheader("Efficiency ratio")
st.line_chart(df.set_index("event_density")[["efficiency_ratio_proxy"]])
st.dataframe(df, use_container_width=True)
st.caption("Simulation operation proxies only; not physical energy measurements.")
