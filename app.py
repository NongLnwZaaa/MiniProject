import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 0. การตั้งค่าหน้าเว็บ (Page Configuration)
# ==========================================
st.set_page_config(page_title="House Price Prediction", page_icon="🏡", layout="wide")

# แทรก Custom CSS เพื่อปรับแต่ง UI ให้ดู Modern
st.markdown("""
<style>
    /* ปรับแต่งปุ่มกดให้ออกแบบแนว Modern & Responsive */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
        color: #f1f1f1;
    }
    /* ปรับแต่งส่วนหัวของ Tabs ให้ชัดเจนขึ้น */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ข้อมูลผู้พัฒนา (แสดงที่ Sidebar)
# ==========================================
with st.sidebar:
    st.title("👨‍💻 ข้อมูลผู้พัฒนา")
    st.image("https://scontent.fbkk31-2.fna.fbcdn.net/v/t39.30808-6/397864360_2110730222603303_7087100876165517819_n.jpg?stp=dst-jpg_tt6&cstp=mx959x960&ctp=s959x960&_nc_cat=106&_nc_map=urlgen_bucketless&ccb=1-7&_nc_sid=6ee11a&_nc_eui2=AeE6k_lewxt-HkAVJvc_EYEyJznkhh0orF4nOeSGHSisXk6IGg4fpAHXOiU0ozERJUlZaUKvuZWJFTr74mY7a4y7&_nc_ohc=LroG0wO-Jl0Q7kNvwH4VSCK&_nc_oc=AdqMYRVra-o0aQKil4i4QNWh2x2BlzKLbkromMvqnb4gG3MA-yC8Jly5vJ4WwQ1hDGoSI6zBAgic_Fzbt-U-kjLY&_nc_zt=23&_nc_ht=scontent.fbkk31-2.fna&_nc_gid=jQF7Ok1fF6BSrtFFgH9-qQ&_nc_ss=7b2a8&oh=00_AQHMu0fPQcDEMYDROMresmDVMKsCzjRnWy6QAKX7TPt71A&oe=6A886C95", width=150) 
    
    st.markdown("""
    **รหัสประจำตัว:** 664245016  
    **ชื่อ-นามสกุล:** นาย กรภัทร์ ถิ่นผาแดง  
    **หมู่เรียน:** 66/43  
    """)
    st.divider()
    st.info("📌 โปรเจ็กต์นี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (30 คะแนน)")

# ==========================================
# 1. โหลดและจัดการข้อมูล (Data Preprocessing)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ 'train.csv' กรุณาตรวจสอบให้แน่ใจว่าไฟล์อยู่ในโฟลเดอร์เดียวกับโค้ด")
        return None, None, None, None, None, None, None, None

    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    target = 'SalePrice'
    
    df_selected = df[features + [target]].dropna()
    X = df_selected[features]
    y = df_selected[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, df_selected, X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

df_raw, df_processed, X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

# ==========================================
# 2. สร้างและเทรนโมเดล (Model Training)
# ==========================================
@st.cache_resource
def train_models(X_train, y_train):
    if X_train is None:
        return None, None
        
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

lr_model, rf_model = train_models(X_train, y_train)

# ==========================================
# 3. ส่วนแสดงผลบนเว็บ (Web Interface - Tabs)
# ==========================================
st.title("🏡 House Price Prediction (Ames Dataset)")
st.markdown("ระบบพยากรณ์ราคาอสังหาริมทรัพย์ด้วย Machine Learning 🤖")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 1. การกำหนดปัญหา", 
    "⚙️ 2. Data Preprocessing", 
    "🧠 3. โมเดล ML", 
    "📊 4. ประเมินผลโมเดล", 
    "🚀 5. ใช้งานแอป"
])

# ------------------------------------------
# Tab 1: การกำหนดปัญหาและ Dataset
# ------------------------------------------
with tab1:
    st.header("1. การกำหนดปัญหาและ Dataset")
    st.markdown("""
    ### 🎯 การกำหนดปัญหา (Problem Statement)
    ในการซื้อขายอสังหาริมทรัพย์ การประเมินราคาบ้านให้เหมาะสมเป็นเรื่องที่ท้าทาย โปรเจ็กต์นี้จึงมุ่งเน้น **สร้างโมเดล Machine Learning เพื่อทำนายราคาขายบ้าน (SalePrice)** ได้อย่างแม่นยำ จากปัจจัยแวดล้อมต่างๆ 

    ### 📊 Dataset: ทำไมถึงเลือกข้อมูลชุดนี้?
    เราเลือกใช้ชุดข้อมูล **Ames Housing Dataset** (`train.csv`) เนื่องจากมีฟีเจอร์ให้พิจารณาถึง 79 ตัวแปร ข้อมูลมีความสมจริง และเป็นโจทย์คลาสสิกที่เหมาะสำหรับการเปรียบเทียบประสิทธิภาพของอัลกอริทึม
    """)
    if df_raw is not None:
        with st.expander("🔎 คลิกเพื่อดูตัวอย่างชุดข้อมูลดิบ (Raw Data)"):
            st.dataframe(df_raw.head(10), use_container_width=True)

# ------------------------------------------
# Tab 2: Data Preprocessing
# ------------------------------------------
with tab2:
    st.header("2. Data Preprocessing")
    st.markdown("""
    เพื่อให้โมเดลทำงานได้อย่างมีประสิทธิภาพและประมวลผลบนเว็บแอปพลิเคชันได้รวดเร็ว ได้มีการเตรียมข้อมูลดังนี้:
    
    *   **🎯 Feature Selection:** เลือกเฉพาะตัวแปรสำคัญ 6 ตัว ได้แก่ `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `FullBath`, `YearBuilt`
    *   **🧹 Handling Missing Values:** กรองเอาเฉพาะข้อมูลที่สมบูรณ์ด้วยคำสั่ง `.dropna()`
    *   **⚖️ Feature Scaling:** ใช้ `StandardScaler` แปลงสเกลข้อมูลให้มีค่าเฉลี่ยเป็น 0 และส่วนเบี่ยงเบนมาตรฐานเป็น 1 
    """)
    
    if df_raw is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("📌 **ข้อมูลก่อนปรับสเกล (Raw Features)**")
            st.dataframe(df_processed[feature_names].head(), use_container_width=True)
        with col2:
            st.write("✨ **ข้อมูลหลังปรับสเกล (Scaled Features)**")
            st.dataframe(pd.DataFrame(X_train, columns=feature_names).head(), use_container_width=True)

# ------------------------------------------
# Tab 3: ทฤษฎีของโมเดล ML
# ------------------------------------------
with tab3:
    st.header("3. การสร้างโมเดล ML และทฤษฎี")
    col1, col2 = st.columns(2)
    with col1:
        st.info("### 📈 Linear Regression\n**ทฤษฎี:** เป็นโมเดลพื้นฐานที่สุด ทำงานโดยพยายามหาเส้นตรงที่ลากผ่านจุดข้อมูลต่างๆ โดยให้ผลรวมของ Error มีค่าน้อยที่สุด\n\n**ข้อดี:** ตีความหมายง่าย เทรนได้รวดเร็ว")
    with col2:
        st.success("### 🌳 Random Forest Regressor\n**ทฤษฎี:** เป็น Ensemble Learning ทำงานโดยสร้าง Decision Tree ขึ้นมาหลายๆ ต้นแบบสุ่ม แล้วนำผลลัพธ์มาหาค่าเฉลี่ย\n\n**ข้อดี:** มีความแม่นยำสูง จัดการกับข้อมูลซับซ้อนได้ดี")

# ------------------------------------------
# Tab 4: การประเมินและเปรียบเทียบ
# ------------------------------------------
with tab4:
    st.header("4. การประเมินและเปรียบเทียบโมเดล")
    
    if lr_model and rf_model:
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # จัด Layout Dashboard ด้วย st.metric แบบ Modern
        st.subheader("📊 สรุปประสิทธิภาพโมเดล")
        met1, met2, met3, met4 = st.columns(4)
        met1.metric(label="Linear Regression (RMSE)", value=f"{lr_rmse:,.0f}")
        met2.metric(label="Linear Regression (R²)", value=f"{lr_r2:.4f}")
        met3.metric(label="Random Forest (RMSE)", value=f"{rf_rmse:,.0f}")
        met4.metric(label="Random Forest (R²)", value=f"{rf_r2:.4f}")
        
        st.divider()
        
        # ตกแต่งกราฟให้คลีนขึ้นด้วย Seaborn whitegrid
        sns.set_theme(style="whitegrid")
        
        results_df = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest Regressor"],
            "RMSE": [lr_rmse, rf_rmse]
        })
        
        col_chart1, col_chart2 = st.columns([1, 1.5])
        
        with col_chart1:
            st.write("📉 **กราฟเปรียบเทียบ RMSE**")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.barplot(x="Model", y="RMSE", data=results_df, ax=ax, palette="mako")
            ax.set_title("Comparison of RMSE between Models", pad=15)
            ax.set_xlabel("Model")
            ax.set_ylabel("RMSE")
            st.pyplot(fig)
            
        with col_chart2:
            st.write("🎯 **กราฟ Actual vs Predicted**")
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
            
            ax1.scatter(y_test, lr_pred, alpha=0.5, color='#4b6cb7', edgecolor='w')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual Price')
            ax1.set_ylabel('Predicted Price')
            ax1.set_title('Linear Regression')
            
            ax2.scatter(y_test, rf_pred, alpha=0.5, color='#182848', edgecolor='w')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual Price')
            ax2.set_title('Random Forest')
            
            plt.tight_layout()
            st.pyplot(fig2)

# ------------------------------------------
# Tab 5: Streamlit Application (ระบบใช้งานจริง)
# ------------------------------------------
with tab5:
    st.header("5. ทดลองประเมินราคาบ้าน")
    st.markdown("กรอกข้อมูลรายละเอียดของบ้านด้านล่าง เพื่อให้โมเดล **Random Forest** ช่วยคาดการณ์ราคาให้คุณ")
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🛠️ สภาพและขนาดพื้นที่**")
            input_qual = st.slider("คุณภาพและวัสดุโดยรวม (OverallQual 1-10)", 1, 10, 5)
            input_area = st.number_input("พื้นที่ใช้สอยเหนือดิน (GrLivArea sq.ft)", min_value=300, max_value=6000, value=1500, step=100)
            input_bsmt = st.number_input("พื้นที่ชั้นใต้ดิน (TotalBsmtSF sq.ft)", min_value=0, max_value=4000, value=1000, step=100)
            
        with col2:
            st.markdown("**🏠 รายละเอียดเพิ่มเติม**")
            input_cars = st.number_input("จำนวนจอดรถในโรงรถ (GarageCars)", min_value=0, max_value=5, value=2)
            input_bath = st.number_input("จำนวนห้องน้ำเต็มรูปแบบ (FullBath)", min_value=1, max_value=4, value=2)
            input_year = st.slider("ปีที่สร้างบ้าน (YearBuilt)", 1872, 2010, 2000)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("💰 ทำนายราคาบ้าน")
        
    if submitted:
        if rf_model is not None:
            new_data = np.array([[input_qual, input_area, input_cars, input_bsmt, input_bath, input_year]])
            new_data_scaled = scaler.transform(new_data)
            predicted_price = rf_model.predict(new_data_scaled)[0]
            
            st.success("🎉 คาดการณ์ราคาบ้านสำเร็จ!")
            st.metric(label="ราคาบ้านที่คาดการณ์ (USD)", value=f"${predicted_price:,.2f}")
            st.balloons()
        else:
            st.error("โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบชุดข้อมูล")