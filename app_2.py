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

# 1. إعدادات الصفحة
st.set_page_config(page_title="توقعات الحديد 16مم", layout="wide")
st.title("🏗️ لوحة تحليل وتوقع أسعار الحديد (16مم)")

DATA_FILE = "iron_16mm_data.csv"

# 2. دالة تنظيف وتجهيز البيانات
def clean_data(df_raw):
    try:
        # تحويل البيانات من عرضي لطولي
        df = df_raw.transpose().reset_index()
        df.columns = ['ds', 'y']
        
        # قاموس الشهور العربية
        months = {'يناير': 'Jan', 'فبراير': 'Feb', 'مارس': 'Mar', 'أبريل': 'Apr', 'ابريل': 'Apr',
                  'مايو': 'May', 'يونيو': 'Jun', 'يوليو': 'Jul', 'أغسطس': 'Aug', 'اغسطس': 'Aug',
                  'سبتمبر': 'Sep', 'أكتوبر': 'Oct', 'اكتوبر': 'Oct', 'نوفمبر': 'Nov', 'ديسمبر': 'Dec'}
        
        # استبدال الشهور العربية
        for ar, en in months.items():
            df['ds'] = df['ds'].astype(str).str.replace(ar, en, regex=False)
        
        # تحويل التواريخ (مع تجاهل الأخطاء مثل كلمة Indicator)
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        
        # تنظيف عمود السعر (إزالة أي نصوص والإبقاء على الأرقام فقط)
        df['y'] = df['y'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        
        # حذف الصفوف الفارغة أو غير الصالحة
        return df.dropna().sort_values('ds')
    except Exception as e:
        st.error(f"خطأ أثناء معالجة البيانات: {e}")
        return None

# 3. دالة السحب (Robust Version for Streamlit Cloud)
def scrape_data():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    try:
        service = None
        # التحقق الذكي: هل نحن على سيرفر لينكس (Streamlit Cloud)؟
        if os.path.exists("/usr/bin/chromium"):
            options.binary_location = "/usr/bin/chromium"
        
        # تحديد مكان الـ Driver
        if os.path.exists("/usr/bin/chromedriver"):
            service = Service("/usr/bin/chromedriver")
        else:
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)
        
        # الرابط
        driver.get("https://www.capmas.gov.eg/data/mainSubject/1/subSubject/13/data-visualization/4274")
        
        # انتظار التحميل
        import time
        time.sleep(15) 
        
        dfs = pd.read_html(driver.page_source)
        driver.quit()
        
        if dfs:
            # حفظ الملف الجديد
            dfs[0].to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
            
    except Exception as e:
        st.sidebar.error(f"حدث خطأ في الاتصال بالمصدر: {e}")
    return False

# --- 4. القائمة الجانبية (Sidebar) ---
st.sidebar.header("لوحة التحكم")

# >> الميزة الجديدة: التحكم في مدة التوقع <<
forecast_days = st.sidebar.slider(
    "حدد مدة التوقع (بالأيام):", 
    min_value=30, 
    max_value=730, 
    value=365, 
    step=30
)

st.sidebar.markdown("---")

# زر التحديث
if st.sidebar.button("تحديث البيانات من المصدر 🔄"):
    with st.sidebar.status("جاري سحب البيانات الجديدة..."):
        if scrape_data():
            st.sidebar.success("تم تحديث الأسعار بنجاح!")
            st.rerun()
        else:
            st.sidebar.error("فشل التحديث. حاول مرة أخرى.")

# --- 5. التشغيل الرئيسي ---
if os.path.exists(DATA_FILE):
    # قراءة الملف
    raw_data = pd.read_csv(DATA_FILE)
    df_clean = clean_data(raw_data)
    
    if df_clean is not None and not df_clean.empty:
        # تدريب النموذج
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        # استخدام المدة التي اختارها المستخدم
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)
        
        # الرسم البياني
        st.subheader(f"📈 رسم بياني لتوقعات الـ {forecast_days} يوم القادمة")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        
        # تحليل المكونات (الاتجاه العام)
        with st.expander("عرض تفاصيل الاتجاه العام (Trend)"):
            fig2 = m.plot_components(forecast)
            st.pyplot(fig2)
        
        # عرض آخر أسعار مسجلة
        st.markdown("### 📋 آخر الأسعار المسجلة فعلياً")
        st.dataframe(df_clean.tail(10).style.format({"y": "{:.2f}"}))
        
        # زر تحميل التوقعات
        csv_exp = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="تحميل ملف التوقعات (CSV)",
            data=csv_exp,
            file_name='forecast_iron_16mm.csv',
            mime='text/csv',
        )
    else:
        st.error("البيانات الموجودة في الملف غير صالحة. برجاء الضغط على 'تحديث البيانات'.")
else:
    st.warning("⚠️ لم يتم العثور على ملف بيانات.")
    st.info("اضغط على زر 'تحديث البيانات من المصدر' في القائمة الجانبية لبدء العمل.")
