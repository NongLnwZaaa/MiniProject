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

# ==========================================
# ข้อมูลผู้พัฒนา (แสดงที่ Sidebar)
# ==========================================
st.sidebar.title("👨‍💻 ข้อมูลผู้พัฒนา")
# ใส่ URL รูปภาพของคุณ หรือใช้ st.image("ชื่อไฟล์รูป.jpg") ถ้ามีไฟล์รูปในโฟลเดอร์เดียวกัน
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150) 
st.sidebar.markdown("""
**รหัสประจำตัว:** 6X1234567  
**ชื่อ-นามสกุล:** นาย สมมติ รักเรียน  
**หมู่เรียน:** 6X/45  
""")
st.sidebar.divider()
st.sidebar.info("โปรเจ็กต์นี้เป็นส่วนหนึ่งของรายวิชา Machine Learning (30 คะแนน)")

# ==========================================
# 1. โหลดและจัดการข้อมูล (Data Preprocessing)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    # โหลดข้อมูล
    try:
        df = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error("ไม่พบไฟล์ 'train.csv' กรุณาตรวจสอบให้แน่ใจว่าไฟล์อยู่ในโฟลเดอร์เดียวกับโค้ด")
        return None, None, None, None, None, None, None

    # เลือกฟีเจอร์เด่นๆ เพื่อความเรียบง่ายและลดความซับซ้อนในการกรอกข้อมูลบนเว็บ
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    target = 'SalePrice'
    
    # จัดการ Missing Values (Drop แถวที่มีค่าว่างในฟีเจอร์ที่เลือก)
    df_selected = df[features + [target]].dropna()
    
    X = df_selected[features]
    y = df_selected[target]
    
    # แบ่งข้อมูล Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ทำ Data Scaling ด้วย StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return df, df_selected, X_train_scaled, X_test_scaled, y_train, y_test, scaler, features

# โหลดข้อมูล
df_raw, df_processed, X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess_data()

# ==========================================
# 2. สร้างและเทรนโมเดล (Model Training)
# ==========================================
@st.cache_resource
def train_models(X_train, y_train):
    if X_train is None:
        return None, None
        
    # โมเดลที่ 1: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # โมเดลที่ 2: Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return lr_model, rf_model

lr_model, rf_model = train_models(X_train, y_train)

# ==========================================
# 3. ส่วนแสดงผลบนเว็บ (Web Interface - Tabs)
# ==========================================
st.title("🏡 House Price Prediction (Ames Dataset)")

# สร้าง Tabs สำหรับเนื้อหา 5 หัวข้อตามที่อาจารย์สั่ง
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. การกำหนดปัญหา", 
    "2. Data Preprocessing", 
    "3. โมเดล ML", 
    "4. ประเมินผลโมเดล", 
    "5. ใช้งานแอป (Streamlit)"
])

# ------------------------------------------
# Tab 1: การกำหนดปัญหาและ Dataset
# ------------------------------------------
with tab1:
    st.header("1. การกำหนดปัญหาและ Dataset (5 คะแนน)")
    st.markdown("""
    ### 🎯 การกำหนดปัญหา (Problem Statement)
    ในการซื้อขายอสังหาริมทรัพย์ การประเมินราคาบ้านให้เหมาะสมเป็นเรื่องที่ท้าทาย เนื่องจากราคาถูกกำหนดโดยปัจจัยหลายอย่าง เช่น ขนาดพื้นที่ จำนวนห้อง ปีที่สร้าง ฯลฯ 
    โปรเจ็กต์นี้จึงมุ่งเน้น **สร้างโมเดล Machine Learning เพื่อทำนายราคาขายบ้าน (SalePrice)** ได้อย่างแม่นยำ จากปัจจัยแวดล้อมต่างๆ 

    ### 📊 Dataset: ทำไมถึงเลือกข้อมูลชุดนี้?
    เราเลือกใช้ชุดข้อมูล **Ames Housing Dataset** (`train.csv`) เนื่องจาก:
    1. **ความท้าทาย:** มีตัวแปรอิสระ (Features) ให้พิจารณาถึง 79 ตัวแปร ทำให้สามารถทดสอบกระบวนการเลือกฟีเจอร์ (Feature Selection) ได้ดี
    2. **สะท้อนโลกจริง:** ข้อมูลมีความสมจริง มีทั้งข้อมูลสูญหาย (Missing values) และมีสเกลข้อมูลที่แตกต่างกัน ทำให้ได้ฝึกฝนทักษะการทำ Data Preprocessing อย่างเต็มที่
    3. **โจทย์คลาสสิก:** เป็นโจทย์ประเภท Regression ที่เหมาะสำหรับการเปรียบเทียบประสิทธิภาพของอัลกอริทึมต่างๆ ได้อย่างชัดเจน
    """)
    if df_raw is not None:
        st.write("**ตัวอย่างข้อมูลดิบ:**")
        st.dataframe(df_raw.head())

# ------------------------------------------
# Tab 2: Data Preprocessing
# ------------------------------------------
with tab2:
    st.header("2. Data Preprocessing (5 คะแนน)")
    st.markdown("""
    เพื่อให้โมเดลสามารถเรียนรู้ได้อย่างมีประสิทธิภาพ ได้มีการเตรียมข้อมูลดังนี้:
    
    1. **การเลือกตัวแปร (Feature Selection):** 
       เพื่อไม่ให้โมเดลซับซ้อนเกินไป และเหมาะสมกับการใช้งานบนเว็บแอปพลิเคชัน จึงได้คัดเลือกฟีเจอร์ที่มีความสัมพันธ์สูงกับราคาบ้านมา 6 ตัว ได้แก่
       - `OverallQual`: คุณภาพวัสดุและการตกแต่งโดยรวม (1-10)
       - `GrLivArea`: พื้นที่ใช้สอยเหนือระดับผิวดิน (ตารางฟุต)
       - `GarageCars`: ขนาดโรงรถ (จำนวนรถที่จอดได้)
       - `TotalBsmtSF`: พื้นที่ชั้นใต้ดินทั้งหมด (ตารางฟุต)
       - `FullBath`: จำนวนห้องน้ำเต็มรูปแบบ
       - `YearBuilt`: ปีที่สร้างบ้าน

    2. **การจัดการค่าว่าง (Handling Missing Values):**
       ทำการกรองเอาเฉพาะข้อมูลที่สมบูรณ์ โดยใช้คำสั่ง `.dropna()` เพื่อลบแถวที่อาจมีค่าข้อมูลสูญหายในฟีเจอร์ที่เราเลือก

    3. **การปรับสเกลข้อมูล (Feature Scaling / Standardization):**
       ข้อมูลแต่ละตัวแปรมีหน่วยต่างกัน (เช่น ปีที่สร้างมีค่าหลักพัน, คุณภาพมีค่า 1-10, พื้นที่มีค่าหลักพัน) จึงใช้ **`StandardScaler`** เพื่อแปลงข้อมูลให้มีค่าเฉลี่ย (Mean) เท่ากับ 0 และส่วนเบี่ยงเบนมาตรฐาน (SD) เท่ากับ 1 ป้องกันไม่ให้โมเดลเอนเอียงไปยังตัวเลขที่มีสเกลใหญ่กว่า
    """)
    
    if df_raw is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("**ข้อมูลก่อนทำ Scaling (X_train)**")
            st.dataframe(df_processed[feature_names].head())
        with col2:
            st.write("**ข้อมูลหลังทำ Scaling (X_train_scaled)**")
            st.dataframe(pd.DataFrame(X_train, columns=feature_names).head())

# ------------------------------------------
# Tab 3: ทฤษฎีของโมเดล ML
# ------------------------------------------
with tab3:
    st.header("3. การสร้างโมเดล ML และทฤษฎี (5 คะแนน)")
    st.markdown("""
    ในโปรเจ็กต์นี้ เราได้สร้างและเปรียบเทียบโมเดลประเภท Regression 2 แบบ ได้แก่:

    ### 📈 1. Linear Regression (การถดถอยเชิงเส้น)
    * **ทฤษฎี:** เป็นโมเดลพื้นฐานที่สุด ทำงานโดยพยายามหาเส้นตรง (หรือระนาบ ในกรณีที่มีหลายตัวแปร) ที่ลากผ่านจุดข้อมูลต่างๆ โดยให้ผลรวมของ "ระยะห่างระหว่างจุดข้อมูลจริงกับเส้นทำนาย" (Error) มีค่าน้อยที่สุด 
    * **สมการ:** $Y = a + b_1X_1 + b_2X_2 + ... + b_nX_n$
    * **ข้อดี:** ตีความหมายง่าย เทรนได้รวดเร็ว เหมาะกับข้อมูลที่มีความสัมพันธ์เป็นเส้นตรง

    ### 🌳 2. Random Forest Regressor (ป่าสุ่ม)
    * **ทฤษฎี:** เป็นโมเดลประเภท Ensemble Learning ทำงานโดยการสร้าง Decision Tree (ต้นไม้ตัดสินใจ) ขึ้นมาหลายๆ ต้น โดยแต่ละต้นจะถูกสร้างจากชุดข้อมูลที่ถูกสุ่มขึ้นมาแบบสุ่ม (Bootstrap) จากนั้นจะนำผลลัพธ์การทำนายของทุกต้นมาหาค่าเฉลี่ยเพื่อให้ได้ผลลัพธ์สุดท้าย
    * **ข้อดี:** มีความแม่นยำสูงมาก จัดการกับความสัมพันธ์ที่ซับซ้อน (Non-linear) ได้ดี และช่วยลดปัญหา Overfitting ที่มักเกิดกับการใช้ Decision Tree ต้นเดียว
    """)

# ------------------------------------------
# Tab 4: การประเมินและเปรียบเทียบ
# ------------------------------------------
with tab4:
    st.header("4. การประเมินและเปรียบเทียบโมเดล (5 คะแนน)")
    
    if lr_model and rf_model:
        # ทำนายผล
        lr_pred = lr_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        # คำนวณ Error
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        
        # สร้างตารางเปรียบเทียบ
        st.subheader("📊 ตารางเปรียบเทียบประสิทธิภาพ")
        results_df = pd.DataFrame({
            "โมเดล": ["Linear Regression", "Random Forest Regressor"],
            "RMSE (ยิ่งต่ำยิ่งดี)": [lr_rmse, rf_rmse],
            "R² Score (ยิ่งเข้าใกล้ 1 ยิ่งดี)": [lr_r2, rf_r2]
        })
        st.table(results_df.style.format({"RMSE (ยิ่งต่ำยิ่งดี)": "{:,.2f}", "R² Score (ยิ่งเข้าใกล้ 1 ยิ่งดี)": "{:.4f}"}))
        
        # กราฟเปรียบเทียบ
        st.subheader("📉 กราฟเปรียบเทียบ RMSE")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x="โมเดล", y="RMSE (ยิ่งต่ำยิ่งดี)", data=results_df, ax=ax, palette="viridis")
        plt.title("Comparison of RMSE between Models")
        st.pyplot(fig)
        
        # กราฟ Actual vs Predicted
        st.subheader("🎯 กราฟเปรียบเทียบค่าจริง (Actual) และค่าที่ทำนาย (Predicted)")
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.scatter(y_test, lr_pred, alpha=0.5, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Price')
        ax1.set_ylabel('Predicted Price')
        ax1.set_title('Linear Regression')
        
        ax2.scatter(y_test, rf_pred, alpha=0.5, color='green')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.set_title('Random Forest')
        
        st.pyplot(fig2)

# ------------------------------------------
# Tab 5: Streamlit Application (ระบบใช้งานจริง)
# ------------------------------------------
with tab5:
    st.header("5. Streamlit Application (5 คะแนน)")
    st.markdown("ทดลองกรอกข้อมูลของบ้านเพื่อทำนายราคาขาย (ใช้อัลกอริทึม **Random Forest** ที่มีความแม่นยำสูงกว่า)")
    
    # แบบฟอร์มรับข้อมูล
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            input_qual = st.slider("คุณภาพและวัสดุโดยรวม (OverallQual 1-10)", 1, 10, 5)
            input_area = st.number_input("พื้นที่ใช้สอยเหนือดิน (GrLivArea sqft)", min_value=300, max_value=6000, value=1500)
            input_cars = st.number_input("จำนวนจอดรถในโรงรถ (GarageCars)", min_value=0, max_value=5, value=2)
            
        with col2:
            input_bsmt = st.number_input("พื้นที่ชั้นใต้ดิน (TotalBsmtSF sqft)", min_value=0, max_value=4000, value=1000)
            input_bath = st.number_input("จำนวนห้องน้ำเต็มรูปแบบ (FullBath)", min_value=1, max_value=4, value=2)
            input_year = st.slider("ปีที่สร้างบ้าน (YearBuilt)", 1872, 2010, 2000)
            
        submitted = st.form_submit_button("💰 ทำนายราคาบ้าน")
        
    if submitted:
        if rf_model is not None:
            # สร้าง Array ข้อมูลใหม่
            new_data = np.array([[input_qual, input_area, input_cars, input_bsmt, input_bath, input_year]])
            # Transform ข้อมูลใหม่ด้วย Scaler ที่เทรนไว้
            new_data_scaled = scaler.transform(new_data)
            # ทำนายราคา
            predicted_price = rf_model.predict(new_data_scaled)[0]
            
            st.success(f"### ราคาบ้านที่คาดการณ์: **${predicted_price:,.2f}**")
            st.balloons()
        else:
            st.error("โมเดลยังไม่พร้อมใช้งาน กรุณาตรวจสอบชุดข้อมูล")