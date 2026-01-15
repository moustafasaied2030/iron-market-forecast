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

# إعدادات واجهة الموقع
st.set_page_config(page_title="توقعات الحديد 16مم", layout="wide")
st.title("🏗️ نظام تحليل وتوقع أسعار الحديد (16مم)")

DATA_FILE = "iron_16mm_data.csv"

# دالة تنظيف البيانات
def clean_data(df_raw):
    try:
        # قلب الجدول وتحويله لصيغة طولية
        df = df_raw.transpose().reset_index()
        df.columns = ['ds', 'y']
        
        # ترجمة الشهور العربية لإنجليزية (Unicode Safe)
        months = {'يناير': 'Jan', 'فبراير': 'Feb', 'مارس': 'Mar', 'أبريل': 'Apr', 'ابريل': 'Apr',
                  'مايو': 'May', 'يونيو': 'Jun', 'يوليو': 'Jul', 'أغسطس': 'Aug', 'سبتمبر': 'Sep',
                  'أكتوبر': 'Oct', 'نوفمبر': 'Nov', 'ديسمبر': 'Dec'}
        
        for ar, en in months.items():
            df['ds'] = df['ds'].astype(str).str.replace(ar, en, regex=False)
        
        # تحويل التاريخ مع معالجة الأخطاء (مثل كلمة Indicator)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        
        # تنظيف الأسعار (إبقاء الأرقام فقط)
        df['y'] = df['y'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        return df.dropna().sort_values('ds')
    except Exception as e:
        st.error(f"خطأ في معالجة البيانات: {e}")
        return None

# دالة السحب (Scraping)
def scrape_data():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://www.capmas.gov.eg/data/mainSubject/1/subSubject/13/data-visualization/4274")
        import time; time.sleep(12) # وقت للتحميل
        dfs = pd.read_html(driver.page_source)
        driver.quit()
        if dfs:
            dfs[0].to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.sidebar.error(f"فشل السحب: {e}")
    return False

# القائمة الجانبية
st.sidebar.header("التحكم في البيانات")
if st.sidebar.button("تحديث البيانات من المصدر 🔄"):
    with st.sidebar.status("جاري التحديث..."):
        if scrape_data():
            st.sidebar.success("تم التحديث!")
            st.rerun()

# العرض الرئيسي
if os.path.exists(DATA_FILE):
    raw = pd.read_csv(DATA_FILE)
    df_final = clean_data(raw)
    
    if df_final is not None:
        # تشغيل Prophet
        m = Prophet(daily_seasonality=True)
        m.fit(df_final)
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        # الرسم البياني
        st.subheader("توقعات الـ 12 شهراً القادمة")
        fig = m.plot(forecast)
        st.pyplot(fig)
        
        st.subheader("البيانات الحالية")
        st.dataframe(df_final.tail(10))
else:
    st.warning("برجاء الضغط على زر التحديث لجلب البيانات لأول مرة.")