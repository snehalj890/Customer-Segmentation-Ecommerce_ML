# 🛍️ Customer Segmentation for E-Commerce Personalization Engine

An end-to-end Machine Learning and full-stack web application designed to automatically segment e-commerce customers based on behavioral, transactional, and demographic metrics—driving targeted marketing strategies and personalized user experiences.

---

## 🚀 Project Overview
Traditional customer segmentation often relies solely on basic demographic data. This project builds an advanced unsupervised learning pipeline that combines demographic baselines with transaction history and pricing sensitivity to categorize users into 5 actionable business segments. The system features both a robust Python backend pipeline (`segmentation.py`) and an interactive **Streamlit Web Application** (`app.py`) for real-time customer simulation and cluster mapping.

---

## 📊 The 9 Advanced Features Engineered
Our model evaluates customers across 4 core e-commerce dimensions:
1. **Demographics:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`
2. **Engineered Ratios:** `Income_to_Spending_Ratio` (balances earnings vs platform spending)
3. **Transactional Metrics:** `Total_Orders` (separates one-time buyers from repeat shoppers), `Average_Order_Value` ($)
4. **Behavioral & Pricing Sensitivity:** `Discount_Sensitivity` (tracks promo code usage/bargain hunting), `Cart_Abandonment_Rate`
5. **Temporal & Loyalty:** `Tenure_Months` (account maturity and long-term engagement)

---

## 💡 E-Commerce Business Segments & Targeted Strategies
| Segment | Profile Focus | Automated Personalization Strategy |
| :--- | :--- | :--- |
| **High Value Customers** | High income, high spend, frequent orders | 💎 **VIP Treatment:** Offer early access to luxury product drops, free expedited shipping, and exclusive rewards. |
| **Budget Buyers** | Price-sensitive, high discount usage | 🏷️ **Value Focus:** Send targeted clearance notifications, budget-friendly bundles, and promotional discount alerts. |
| **Impulse Shoppers** | High engagement, quick conversions | ⚡ **Engagement Play:** Trigger flash sales, countdown timers, and gamified cross-sells. |
| **Potential Loyalists** | Developing tenure, steady order frequency | 🤝 **Nurture Campaigns:** Enroll in automated onboarding email series and tiered loyalty points. |
| **Low Engagement Customers** | High cart abandonment, low activity | 🔄 **Win-Back Flows:** Launch 'We Miss You' re-engagement campaigns with personalized browse incentives. |

---

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning & Sci-Py:** Scikit-Learn (K-Means Clustering, StandardScaler, PCA), SciPy (Hierarchical Clustering / Ward Linkage)
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn (2D, 3D, and PCA scatter plots, Dendrograms)
* **Web UI Framework:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ Local Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/snehalj890/Customer-Segmentation-Ecommerce.git](https://github.com/snehalj890/Customer-Segmentation-Ecommerce.git)
   cd Customer-Segmentation-Ecommerce
