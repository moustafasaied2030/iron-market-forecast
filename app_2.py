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

# إعدادات الصفحة
st.set_page_config(page_title="Debug Mode", layout="wide")
st.title("🛠️ وضع إصلاح الأخطاء (Debug Mode)")

DATA_FILE = "iron_16mm_data.csv"

# دالة تنظيف البيانات (مع إظهار الأخطاء)
def clean_data(df_raw):
    try:
        st.write("... جاري تنظيف البيانات ...")
        df = df_raw.transpose().reset_index()
        df.columns = ['ds', 'y']
        
        # طباعة شكل البيانات قبل التنظيف للتأكد
        st.write("شكل البيانات قبل المعالجة:")
        st.write(df.head())

        months = {'يناير': 'Jan', 'فبراير': 'Feb', 'مارس': 'Mar', 'أبريل': 'Apr', 'ابريل': 'Apr',
                  'مايو': 'May', 'يونيو': 'Jun', 'يوليو': 'Jul', 'أغسطس': 'Aug', 'سبتمبر': 'Sep',
                  'أكتوبر': 'Oct', 'نوفمبر': 'Nov', 'ديسمبر': 'Dec'}
        
        for ar, en in months.items():
            df['ds'] = df['ds'].astype(str).str.replace(ar, en, regex=False)
        
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
        
        result = df.dropna().sort_values('ds')
        
        if result.empty:
            st.error("البيانات فارغة بعد التنظيف! تأكد أن التواريخ والأرقام في الملف الأصلي صحيحة.")
            return None
            
        return result
    except Exception as e:
        st.error(f"حدث خطأ داخل دالة التنظيف: {e}")
        return None

# دالة السحب (Scraping) المعدلة للسيرفر
def scrape_data():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    try:
        service = None
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
            dfs[0].to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.error(f"فشل السحب: {e}")
    return False

# --- الكود الرئيسي ---

# 1. فحص وجود الملف
if os.path.exists(DATA_FILE):
    st.success(f"1. تم العثور على الملف: {DATA_FILE}")
    
    try:
        raw_data = pd.read_csv(DATA_FILE)
        st.write("2. تم قراءة الملف بنجاح. عدد الصفوف:", len(raw_data))
        
        # عرض محتوى الملف الخام (للتأكد أنه ليس فارغاً)
        with st.expander("عرض الملف الأصلي"):
            st.dataframe(raw_data)
        
        # محاولة التنظيف
        df_final = clean_data(raw_data)
        
        if df_final is not None:
            st.success(f"3. نجح التنظيف. عدد الصفوف الصالحة: {len(df_final)}")
            
            # تشغيل Prophet
            with st.spinner("جاري تدريب النموذج..."):
                m = Prophet(daily_seasonality=True)
                m.fit(df_final)
                future = m.make_future_dataframe(periods=365)
                forecast = m.predict(future)
            
            st.subheader("الرسم البياني للتوقعات")
            fig = m.plot(forecast)
            st.pyplot(fig)
        else:
            st.error("توقف البرنامج لأن البيانات غير صالحة.")
            
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع: {e}")

else:
    st.warning("⚠️ لم يتم العثور على ملف البيانات (iron_16mm_data.csv).")
    st.info("من فضلك اضغط على زر التحديث في القائمة الجانبية لمحاولة جلبه.")

# القائمة الجانبية
st.sidebar.header("التحكم")
if st.sidebar.button("تحديث البيانات 🔄"):
    with st.sidebar.status("جاري الاتصال بالموقع..."):
        if scrape_data():
            st.sidebar.success("تم التحديث!")
            st.rerun()
        else:
            st.sidebar.error("فشل التحديث.")
