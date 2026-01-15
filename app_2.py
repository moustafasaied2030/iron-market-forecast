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
from datetime import datetime, timedelta

# 1. إعدادات الصفحة
st.set_page_config(page_title="توقعات الحديد 16مم", layout="wide")
st.title("🏗️ لوحة تحليل وتوقع أسعار الحديد (16مم)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة تنظيف البيانات
def clean_data(df_raw):
    try:
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
        st.error(f"خطأ معالجة: {e}")
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
        
        import time
        time.sleep(15) 
        
        dfs = pd.read_html(driver.page_source)
        driver.quit()
        
        if dfs:
            dfs[0].to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
    except Exception as e:
        st.sidebar.error(f"خطأ اتصال: {e}")
    return False

# --- القائمة الجانبية ---
st.sidebar.header("لوحة التحكم")

# أداة 1: تحديد مدة الرسم البياني
forecast_days = st.sidebar.slider("مدة الرسم البياني (أيام):", 30, 730, 365, 30)

st.sidebar.markdown("---")

# أداة 2: البحث عن سعر في تاريخ محدد (جديد!) 📅
st.sidebar.subheader("🔍 استعلام عن تاريخ محدد")
target_date_input = st.sidebar.date_input("اختر التاريخ:", datetime.now())

st.sidebar.markdown("---")

if st.sidebar.button("تحديث البيانات 🔄"):
    with st.sidebar.status("جاري التحديث..."):
        if scrape_data():
            st.sidebar.success("تم!")
            st.rerun()

# --- المحتوى الرئيسي ---
if os.path.exists(DATA_FILE):
    raw_data = pd.read_csv(DATA_FILE)
    df_clean = clean_data(raw_data)
    
    if df_clean is not None and not df_clean.empty:
        # تجهيز النموذج
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        # إنشاء نطاق زمني للمستقبل (بناءً على اختيار المستخدم في الـ Slider)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # --- منطق البحث عن التاريخ المحدد ---
        # نحول التاريخ المختار لصيغة datetime للمقارنة
        target_date = pd.to_datetime(target_date_input)
        last_real_date = df_clean['ds'].max()
        
        # عرض نتيجة البحث في مربع بارز في الأعلى
        st.info(f"📅 التاريخ المختار: {target_date.strftime('%Y-%m-%d')}")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if target_date <= last_real_date:
                # 1. حالة الماضي: البحث في البيانات الحقيقية
                # نبحث عن أقرب تاريخ مسجل (لأن البيانات قد تكون شهرية وليست يومية)
                nearest_date = df_clean.iloc[(df_clean['ds'] - target_date).abs().argsort()[:1]]
                real_price = nearest_date['y'].values[0]
                real_date_found = nearest_date['ds'].dt.strftime('%Y-%m-%d').values[0]
                
                st.metric(label="السعر المسجل (تاريخي)", value=f"{real_price:,.0f} جنيه")
                st.caption(f"* أقرب سجل متوفر كان بتاريخ: {real_date_found}")
            else:
                # 2. حالة المستقبل: استخدام التوقعات
                # نتأكد أن التاريخ المستقبلي موجود ضمن نطاق التوقع
                days_diff = (target_date - last_real_date).days
                if days_diff > forecast_days:
                    st.warning(f"⚠️ هذا التاريخ بعيد جداً! يرجى زيادة 'مدة الرسم البياني' من الشريط الجانبي إلى أكثر من {days_diff} يوم.")
                else:
                    # استخراج التوقع لهذا اليوم
                    pred_row = forecast[forecast['ds'] == target_date]
                    if not pred_row.empty:
                        pred_price = pred_row['yhat'].values[0]
                        lower_price = pred_row['yhat_lower'].values[0]
                        upper_price = pred_row['yhat_upper'].values[0]
                        
                        st.metric(label="السعر المتوقع", value=f"{pred_price:,.0f} جنيه")
                        st.write(f"تتراوح التوقعات بين: **{lower_price:,.0f}** و **{upper_price:,.0f}** جنيه")
                    else:
                        st.warning("يرجى توسيع نطاق التوقع ليشمل هذا التاريخ.")

        # --- الرسم البياني العام ---
        st.divider()
        st.subheader("📈 الرسم البياني للأسعار")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        
        # تحميل البيانات
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("تحميل التوقعات CSV", csv, "forecast.csv", "text/csv")
        
    else:
        st.error("البيانات تالفة.")
else:
    st.warning("لا يوجد ملف بيانات. اضغط تحديث.")
