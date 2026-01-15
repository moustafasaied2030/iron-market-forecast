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
st.set_page_config(page_title="توقعات الحديد 16مم", layout="wide")
st.title("🏗️ نظام تحليل وتوقع أسعار الحديد (16مم)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة تنظيف البيانات
def clean_data(df_raw):
    try:
        # التعامل بذكاء مع البيانات
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
        
        return df.dropna().sort_values('ds')
    except Exception as e:
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

# أداة 1: مدة التوقع (تم إصلاح الخطأ هنا)
forecast_days = st.sidebar.slider("مدة الرسم البياني (أيام):", 30, 730, 365, 30)

st.sidebar.markdown("---")

# أداة 2: البحث عن تاريخ
st.sidebar.subheader("📅 استعلام عن تاريخ")
target_date_input = st.sidebar.date_input("اختر اليوم:", datetime.now())

st.sidebar.markdown("---")

# أداة 3: تصحيح السعر اليدوي
st.sidebar.subheader("🛠️ تصحيح سعر السوق")
st.sidebar.caption("اضبط السعر يدوياً لتحديث النموذج:")
new_price = st.sidebar.number_input("سعر اليوم الفعلي:", value=36000, step=500)

if st.sidebar.button("تسجيل السعر وتحديث 💾"):
    if os.path.exists(DATA_FILE):
        current_df = pd.read_csv(DATA_FILE)
        if 'ds' not in current_df.columns:
            current_df = clean_data(pd.read_csv(DATA_FILE))
            
        today_date = datetime.now().strftime('%Y-%m-%d')
        new_row = pd.DataFrame({'ds': [today_date], 'y': [new_price]})
        
        updated_df = pd.concat([current_df, new_row], ignore_index=True)
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
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # --- منطق عرض التاريخ المحدد ---
        target_date = pd.to_datetime(target_date_input)
        last_real_date = pd.to_datetime(df_clean['ds']).max()
        
        st.info(f"📍 نتيجة البحث عن يوم: {target_date.strftime('%Y-%m-%d')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if target_date <= last_real_date:
                df_clean['ds'] = pd.to_datetime(df_clean['ds'])
                nearest_idx = (df_clean['ds'] - target_date).abs().idxmin()
                real_row = df_clean.loc[nearest_idx]
                
                # (تم إصلاح الخطأ هنا أيضاً)
                st.metric("السعر المسجل (تاريخي)", f"{real_row['y']:,.0f} جنيه")
                formatted_date = real_row['ds'].strftime('%Y-%m-%d')
                st.caption(f"أقرب بيان متوفر: {formatted_date}")
            
            else:
                pred_row = forecast[forecast['ds'] == target_date]
                if not pred_row.empty:
                    p = pred_row.iloc[0]
                    st.metric("السعر المتوقع", f"{p['yhat']:,.0f} جنيه")
                    st.caption(f"المدى المتوقع: {p['yhat_lower']:,.0f} - {p['yhat_upper']:,.0f}")
                else:
                    st.warning("التاريخ خارج النطاق. قم بزيادة المدة.")

        st.divider()
        st.subheader("📈 مسار الأسعار")
        fig = m.plot(forecast)
        st.pyplot(fig)
        
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("تحميل التوقعات CSV", csv, "forecast.csv", "text/csv")
        
    else:
        st.error("البيانات تالفة.")
else:
    st.warning("⚠️ لم يتم العثور على ملف البيانات (iron_16mm_data.csv).")
    st.info("قم برفع الملف إلى GitHub أو اضغط زر التحديث.")
