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
st.title("🏗️ توقعات أسعار الحديد (بيانات من 2020 فقط)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة تنظيف البيانات (مع الفلتر الجديد 2020)
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

        # >>> التعديل الجديد: تصفية البيانات من 2020 فقط <<<
        df_filtered = df[df['ds'] >= '2020-01-01']
        
        return df_filtered

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

forecast_days = st.sidebar.slider("مدة الرسم البياني (أيام):", 30, 730, 365, 30)

st.sidebar.markdown("---")

st.sidebar.subheader("📅 استعلام عن تاريخ")
target_date_input = st.sidebar.date_input("اختر اليوم:", datetime.now())

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ تصحيح سعر السوق")
st.sidebar.caption("اضبط السعر يدوياً لتحديث النموذج:")
new_price = st.sidebar.number_input("سعر اليوم الفعلي:", value=36000, step=500)

if st.sidebar.button("تسجيل السعر وتحديث 💾"):
    if os.path.exists(DATA_FILE):
        current_df = pd.read_csv(DATA_FILE)
        # نحتاج للتنظيف لتوحيد التنسيق قبل الدمج
        # ملاحظة: الدمج يتم على الملف الأصلي، الفلترة تتم عند العرض فقط
        if 'ds' not in current_df.columns:
            # تنظيف مؤقت لغرض الدمج
            temp_df = clean_data(pd.read_csv(DATA_FILE)) # هذا سيرجع بيانات 2020 فقط
            # لكن للحفاظ على البيانات القديمة في الملف (احتياطياً)، سنعتمد على التنسيق البسيط
            pass 
            
        today_date = datetime.now().strftime('%Y-%m-%d')
        new_row = pd.DataFrame({'ds': [today_date], 'y': [new_price]})
        
        # لضمان عدم فساد الملف، نقرأه ونضيف عليه ببساطة
        df_disk = pd.read_csv(DATA_FILE)
        if 'ds' not in df_disk.columns: # لو الملف لسه خام
             # نضطر نعمله تنسيق ونحفظه
             df_disk = clean_data(df_disk) # سيحفظ 2020 فقط وهذا المطلوب
        
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
    df_clean = clean_data(raw_data) # هنا سيتم تطبيق فلتر 2020
    
    if df_clean is not None and not df_clean.empty:
        
        # عرض تنبيه للمستخدم
        min_date = df_clean['ds'].min().date()
        st.info(f"📊 يتم الآن تحليل البيانات بداية من: **{min_date}** (تم استبعاد ما قبل 2020)")

        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # --- منطق البحث ---
        target_date = pd.to_datetime(target_date_input)
        last_real_date = pd.to_datetime(df_clean['ds']).max()
        
        st.divider()
        st.markdown(f"### 📍 نتيجة البحث عن يوم: {target_date.strftime('%Y-%m-%d')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if target_date <= last_real_date:
                # التأكد أن التاريخ المطلوب ليس قبل 2020
                if target_date < pd.to_datetime('2020-01-01'):
                    st.warning("⚠️ التاريخ المختار قبل سنة 2020، والبيانات غير متاحة في هذا النطاق.")
                else:
                    df_clean['ds'] = pd.to_datetime(df_clean['ds'])
                    nearest_idx = (df_clean['ds'] - target_date).abs().idxmin()
                    real_row = df_clean.loc[nearest_idx]
                    
                    st.metric("السعر المسجل (تاريخي)", f"{real_row['y']:,.0f} جنيه")
                    st.caption(f"أقرب بيان: {real_row['ds'].strftime('%Y-%m-%d')}")
            else:
                pred_row = forecast[forecast['ds'] == target_date]
                if not pred_row.empty:
                    p = pred_row.iloc[0]
                    st.metric("السعر المتوقع", f"{p['yhat']:,.0f} جنيه")
                    st.caption(f"المدى: {p['yhat_lower']:,.0f} - {p['yhat_upper']:,.0f}")
                else:
                    st.warning("خارج النطاق.")

        st.subheader("📈 مسار الأسعار (2020 - المستقبل)")
        fig = m.plot(forecast)
        st.pyplot(fig)
        
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button("تحميل التوقعات CSV", csv, "forecast.csv", "text/csv")
        
    else:
        st.error("البيانات فارغة أو لا توجد بيانات بعد 2020.")
else:
    st.warning("⚠️ لم يتم العثور على ملف البيانات.")
