# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 1. إعدادات الصفحة
st.set_page_config(page_title="توقعات الحديد 2020+", layout="wide")
st.title("🏗️ لوحة تحليل أسعار الحديد (بيانات 2020+)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة التنظيف
def clean_data(df_raw):
    try:
        if 'ds' in df_raw.columns and 'y' in df_raw.columns:
            df = df_raw.copy()
        else:
            df = df_raw.transpose().reset_index()
            df.columns = ['ds', 'y']
        
        months = {'يناير': 'Jan', 'فبراير': 'Feb', 'مارس': 'Mar', 'أبريل': 'Apr', 'ابريل': 'Apr',
                  'مايو': 'May', 'يونيو': 'Jun', 'يوليو': 'Jul', 'أغسطس': 'Aug', 'اغسطس': 'Aug',
                  'سبتمبر': 'Sep', 'أكتوبر': 'Oct', 'اكتوبر': 'Oct', 'نوفمبر': 'Nov', 'ديسمبر': 'Dec'}
        
        for ar, en in months.items():
            df['ds'] = df['ds'].astype(str).str.replace(ar, en, regex=False)
        
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = df['y'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        df = df.dropna().sort_values('ds')
        # فلتر 2020
        df_filtered = df[df['ds'] >= '2020-01-01']
        return df_filtered
    except:
        return None

# 3. دالة السحب
def scrape_data():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    try:
        service = None
        if os.path.exists("/usr/bin/chromium"):
            options.binary_location = "/usr/bin/chromium"
        
        if os.path.exists("/usr/bin/chromedriver"):
            service = Service("/usr/bin/chromedriver")
        else:
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.capmas.gov.eg/data/mainSubject/1/subSubject/13/data-visualization/4274")
        
        import time; time.sleep(15) 
        dfs = pd.read_html(driver.page_source)
        driver.quit()
        
        if dfs:
            df_new = dfs[0].transpose().reset_index()
            df_new.columns = ['ds', 'y']
            df_new.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
    return False

# --- القائمة الجانبية ---
st.sidebar.header("🎛️ لوحة التحكم")

forecast_days = st.sidebar.slider("مدة التوقع (أيام):", 30, 730, 365, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 استعلام عن تاريخ")
target_date_input = st.sidebar.date_input("اختر اليوم:", datetime.now())

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ تصحيح سعر السوق")
new_price = st.sidebar.number_input("سعر اليوم الفعلي:", value=36000, step=500)

if st.sidebar.button("تسجيل السعر وتحديث 💾"):
    if os.path.exists(DATA_FILE):
        df_disk = pd.read_csv(DATA_FILE)
        if 'ds' not in df_disk.columns:
             df_disk = clean_data(df_disk)
        
        today_date = datetime.now().strftime('%Y-%m-%d')
        new_row = pd.DataFrame({'ds': [today_date], 'y': [new_price]})
        
        updated_df = pd.concat([df_disk, new_row], ignore_index=True)
        updated_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        st.sidebar.success("تم الحفظ!")
        st.rerun()

st.sidebar.markdown("---")
if st.sidebar.button("سحب بيانات المصدر 🔄"):
    with st.sidebar.status("جاري السحب..."):
        if scrape_data():
            st.sidebar.success("تم!")
            st.rerun()

# --- المحتوى الرئيسي ---
if os.path.exists(DATA_FILE):
    raw_data = pd.read_csv(DATA_FILE)
    df_clean = clean_data(raw_data)
    
    if df_clean is not None and not df_clean.empty:
        
        # تدريب النموذج
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # --- البحث ---
        target_date = pd.to_datetime(target_date_input)
        st.divider()
        st.info(f"📍 سعر يوم: {target_date.strftime('%Y-%m-%d')}")
        
        # منطق السعر (نفس السابق)
        last_real = pd.to_datetime(df_clean['ds']).max()
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            if target_date <= last_real:
                 df_clean['ds'] = pd.to_datetime(df_clean['ds'])
                 real_row = df_clean.loc[(df_clean['ds'] - target_date).abs().idxmin()]
                 st.metric("السعر المسجل", f"{real_row['y']:,.0f}")
            else:
                 pred = forecast[forecast['ds'] == target_date]
                 if not pred.empty:
                     st.metric("السعر المتوقع", f"{pred.iloc[0]['yhat']:,.0f}")
                 else:
                     st.warning("خارج النطاق")

        # --- الرسومات البيانية (الجزء الجديد) ---
        st.divider()
        st.subheader("📊 الرسومات البيانية والتحليل")
        
        # استخدام التبويبات (Tabs) لتنظيم العرض
        tab1, tab2 = st.tabs(["📉 التوقعات الأساسية", "📈 التحليل الفني (Trend & Seasonality)"])
        
        with tab1:
            st.write("الرسم البياني يوضح حركة السعر المتوقعة شاملة كل العوامل:")
            fig1 = m.plot(forecast)
            st.pyplot(fig1)
            
        with tab2:
            st.write("هنا نقوم بتفكيك السعر إلى عوامله الأولية:")
            st.markdown("""
            * **Trend:** الاتجاه العام للسعر (هل هو صاعد أم هابط على المدى الطويل؟).
            * **Weekly:** تأثير أيام الأسبوع (أحياناً غير مؤثر في الحديد).
            * **Yearly (Seasonal):** تأثير شهور السنة (الموسمية - متى يرتفع سنوياً؟).
            """)
            # هذا الأمر يرسم الـ Trend والـ Seasonal
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)

        # تحميل
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("تحميل التوقعات CSV", csv, "forecast.csv", "text/csv")
        
    else:
        st.error("البيانات فارغة.")
else:
    st.warning("لا يوجد ملف بيانات.")
