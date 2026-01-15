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
        # التعامل بذكاء مع البيانات سواء كانت بالعرض (من المصدر) أو بالطول (بعد التعديل اليدوي)
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
            # نحفظ البيانات الخام بشكل يسمح بدمجها لاحقاً
            df_new = dfs[0].transpose().reset_index()
            df_new.columns = ['ds', 'y']
            df_new.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.sidebar.error(f"خطأ اتصال: {e}")
    return False

# --- القائمة الجانبية (كل الأدوات هنا) ---
st.sidebar.header("🎛️ لوحة التحكم")

# أداة 1: مدة التوقع
forecast_days = st.sidebar.slider("مدة الرسم البياني (أ
