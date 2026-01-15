# أضف هذه المكتبات في الأعلى إذا لم تكن موجودة
import os
import shutil

# ... (باقي الكود كما هو) ...

# دالة السحب (Scraping) - النسخة المعدلة للسيرفر
def scrape_data():
    options = Options()
    options.add_argument("--headless") # ضروري جداً للسيرفر
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    # تحسين مهم: تحديد مكان المتصفح والـ Driver يدوياً لضمان التوافق
    # هذا الكود يعمل بذكاء على السيرفر وعلى جهازك الشخصي
    
    try:
        service = None
        
        # 1. محاولة العثور على Chrome في مسار Linux (Streamlit Cloud)
        if os.path.exists("/usr/bin/chromium"):
            options.binary_location = "/usr/bin/chromium"
        
        # 2. محاولة العثور على Driver السيرفر
        if os.path.exists("/usr/bin/chromedriver"):
            service = Service("/usr/bin/chromedriver")
        else:
            # 3. إذا لم نكن على السيرفر (Windows)، نستخدم المدير التلقائي
            service = Service(ChromeDriverManager().install())

        driver = webdriver.Chrome(service=service, options=options)
        
        # بدء التصفح
        driver.get("https://www.capmas.gov.eg/data/mainSubject/1/subSubject/13/data-visualization/4274")
        
        import time
        time.sleep(15) # زيادة وقت الانتظار قليلاً لضمان التحميل
        
        dfs = pd.read_html(driver.page_source)
        driver.quit()
        
        if dfs:
            dfs[0].to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
            return True
            
    except Exception as e:
        st.sidebar.error(f"فشل السحب: {e}")
        # طباعة الخطأ كاملاً في الـ Logs للمساعدة في الحل
        print(f"Full Error: {e}")
        
    return False
