import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Personalization Engine",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.3rem; color: #FF4B4B; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #A0A0A0; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Dataset & Train Model (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("Mall_Customers.csv")
    np.random.seed(42)
    data['Income_to_Spending_Ratio'] = data['Annual Income (k$)'] / (data['Spending Score (1-100)'] + 1)
    data['Total_Orders'] = np.random.randint(1, 20, size=len(data))
    data['Average_Order_Value'] = (data['Annual Income (k$)'] * 500) / (data['Total_Orders'] + 1)
    data['Discount_Sensitivity'] = np.random.uniform(0.0, 1.0, size=len(data))
    data['Cart_Abandonment_Rate'] = np.random.uniform(0.1, 0.9, size=len(data))
    data['Tenure_Months'] = np.random.randint(1, 48, size=len(data))
    
    features = [
        'Age', 'Annual Income (k$)', 'Spending Score (1-100)',
        'Income_to_Spending_Ratio', 'Total_Orders', 'Average_Order_Value',
        'Discount_Sensitivity', 'Cart_Abandonment_Rate', 'Tenure_Months'
    ]
    
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    data['Cluster'] = kmeans.labels_
    return scaler, kmeans, data, features

scaler, kmeans, df_data, feature_cols = load_and_train_model()

segment_metadata = {
    0: {"name": "High Value Customers", "strategy": "💎 **VIP Treatment:** Offer early access to luxury product drops, free expedited shipping, and exclusive rewards."},
    1: {"name": "Budget Buyers", "strategy": "🏷️ **Value Focus:** Send targeted clearance notifications, budget-friendly bundles, and promotional discount alerts."},
    2: {"name": "Impulse Shoppers", "strategy": "⚡ **Engagement Play:** Trigger flash sales, countdown timers, and gamified cross-sells."},
    3: {"name": "Potential Loyalists", "strategy": "🤝 **Nurture Campaigns:** Enroll in automated onboarding email series and tiered loyalty points."},
    4: {"name": "Low Engagement Customers", "strategy": "🔄 **Win-Back Flows:** Launch 'We Miss You' re-engagement campaigns with personalized browse incentives."}
}

# ---------------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------------
st.sidebar.header("🎛️ Customer Profile Inputs")
age = st.sidebar.slider("Age", 18, 70, 25)
income = st.sidebar.slider("Annual Income (k$)", 15, 150, 60)
spending = st.sidebar.slider("Spending Score (1-100)", 1, 100, 80)
total_orders = st.sidebar.slider("Total Orders", 1, 20, 5)
avg_order_val = st.sidebar.slider("Average Order Value ($)", 100, 10000, 3000)
discount_sens = st.sidebar.slider("Discount Sensitivity (0-1)", 0.0, 1.0, 0.4)
cart_abando = st.sidebar.slider("Cart Abandonment Rate (0-1)", 0.0, 1.0, 0.2)
tenure = st.sidebar.slider("Tenure (Months)", 1, 48, 12)

income_spending_ratio = income / (spending + 1)

# ---------------------------------------------------------
# Main Dashboard UI
# ---------------------------------------------------------
st.markdown('<p class="main-header">🛍️ E-Commerce Personalization Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive Customer Segmentation & Behavioral Targeting Dashboard</p>', unsafe_allow_html=True)
st.divider()

# Prediction Computation
input_data = [[
    age, income, spending, income_spending_ratio,
    total_orders, avg_order_val, discount_sens, cart_abando, tenure
]]

scaled_input = scaler.transform(input_data)
pred_cluster = kmeans.predict(scaled_input)[0]
segment_info = segment_metadata[pred_cluster]

# Layout Columns
col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown("### 🎯 Real-Time Classification")
    st.success(f"### Segment: {segment_info['name']}")
    st.info(f"**Cluster ID:** {pred_cluster}")
    
    st.markdown("### 🚀 Personalization Strategy")
    st.markdown(segment_info['strategy'])

with col2:
    st.markdown("### 📊 Input Parameter Breakdown")
    input_df = pd.DataFrame(input_data, columns=feature_cols).T
    input_df.columns = ["Value"]
    st.dataframe(input_df, height=220, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# Interactive Visualizations Section
# ---------------------------------------------------------
st.markdown("### 📈 Customer Distribution & Position Analysis")

tab1, tab2 = st.tabs(["Income vs Spending Cluster Map", "Segment Population Breakdown"])

with tab1:
    st.write("This scatter plot maps all customers by Annual Income vs. Spending Score. The red star **(⭐)** marks where the current simulated customer sits.")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=df_data, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)', 
        hue='Cluster', 
        palette='viridis', 
        s=70, 
        alpha=0.8, 
        ax=ax
    )
    # Plot the user's input point as a prominent red star
    ax.scatter([income], [spending], color='red', s=250, marker='*', label='Current Customer', edgecolor='black')
    ax.set_title("Customer Segmentation Map (Income vs Spending)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

with tab2:
    st.write("Total number of customers belonging to each e-commerce segment across the dataset:")
    cluster_counts = df_data['Cluster'].map(lambda x: segment_metadata[x]['name']).value_counts()
    st.bar_chart(cluster_counts)