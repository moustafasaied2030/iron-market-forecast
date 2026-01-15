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
st.title("🏗️ لوحة تحليل وتوقع أسعار الحديد (16مم)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة التنظيف
def clean_data(df_raw):
    try:
        # إذا كان الملف يحتوي على بيانات مدخلة يدوياً، قد لا يحتاج لـ Transpose
        if 'ds' in df_raw.columns and 'y' in df_raw.columns:
            df = df_raw.copy()
        else:
            df = df_raw.transpose().reset_index()
            df.columns = ['ds', 'y']
        
        # قاموس الشهور
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
            # هنا التعديل: نحفظ الملف بصيغة بسيطة لسهولة الدمج لاحقاً
            df_new = dfs[0].transpose().reset_index()
            df_new.columns = ['ds', 'y']
            df_new.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.sidebar.error(f"خطأ اتصال: {e}")
    return False

# --- القائمة الجانبية ---
st.sidebar.header("لوحة التحكم")

forecast_days = st.sidebar.slider("مدة الرسم البياني (أيام):", 30, 730, 365, 30)

st.sidebar.markdown("---")

# >>> الميزة الجديدة: تصحيح سعر السوق يدوياً <<<
st.sidebar.subheader("🛠️ تصحيح سعر السوق")
st.sidebar.info("إذا كان السعر الرسمي قديماً، أدخل سعر اليوم هنا لتحديث النموذج.")

new_price = st.sidebar.number_input("سعر الطن اليوم (جنيه):", value=36000, step=500)

if st.sidebar.button("تسجيل السعر الحالي وتحديث 💾"):
    if os.path.exists(DATA_FILE):
        # 1. تحميل البيانات القديمة
        current_df = pd.read_csv(DATA_FILE)
        
        # التأكد من توحيد أسماء الأعمدة
        if 'ds' not in current_df.columns:
            # محاولة إصلاح التنسيق إذا كان قادماً من السحب المباشر
            current_df = clean_data(pd.read_csv(DATA_FILE)) # تنظيف مبدئي
            
        # 2. إنشاء صف جديد بتاريخ اليوم
        today_date = datetime.now().strftime('%Y-%m-%d')
        new_row = pd.DataFrame({'ds': [today_date], 'y': [new_price]})
        
        # 3. الدمج
        updated_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # 4. الحفظ
        updated_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
        
        st.sidebar.success(f"تم تسجيل السعر {new_price} بتاريخ {today_date}!")
        st.rerun()
    else:
        st.sidebar.error("لا يوجد ملف بيانات لتحديثه. قم بالسحب أولاً.")

st.sidebar.markdown("---")
# زر التحديث الأصلي
if st.sidebar.button("سحب بيانات جديدة من المصدر 🔄"):
    with st.sidebar.status("جاري التحديث..."):
        if scrape_data():
            st.sidebar.success("تم!")
            st.rerun()

# --- المحتوى الرئيسي ---
if os.path.exists(DATA_FILE):
    raw_data = pd.read_csv(DATA_FILE)
    df_clean = clean_data(raw_data)
    
    if df_clean is not None and not df_clean.empty:
        # عرض آخر سعر مسجل في البيانات
        last_date = df_clean['ds'].max()
        last_price = df_clean.iloc[-1]['y']
        
        st.warning(f"⚠️ آخر بيان مسجل في النظام بتاريخ: **{last_date.date()}** بسعر: **{last_price:,.0f} جنيه**")
        if last_price > 45000:
             st.error("يبدو أن السعر المسجل قديم ويعكس فترة الأزمة. يرجى استخدام 'تصحيح سعر السوق' في القائمة الجانبية لإدخال السعر الحقيقي (36,000 مثلاً).")

        # تجهيز النموذج
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        st.divider()
        st.subheader("📈 مسار الأسعار (مع التحديثات)")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        
    else:
        st.error("البيانات تالفة.")
else:
    st.warning("لا يوجد ملف بيانات.")
