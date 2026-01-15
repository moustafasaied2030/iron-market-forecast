# -*- coding: utf-8 -*-
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# PART 1: SCRAPING (16mm Only)
# ==========================================
def step_1_scrape_16mm():
    print("\n=== STEP 1: STARTING SCRAPER ===")
    
    options = webdriver.ChromeOptions()
    options.add_experimental_option("detach", True)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    url = "https://www.capmas.gov.eg/data/mainSubject/1/subSubject/13/data-visualization/4274"
    driver.get(url)

    print("\n" + "!"*60)
    print("ACTION REQUIRED:")
    print("1. Please Select 'Iron 16mm' manually in the browser.")
    print("2. Select the Years.")
    print("3. Wait for the table.")
    print("!"*60)
    
    input("\n>>> Press ENTER here after the table is visible... ")

    try:
        dfs = pd.read_html(driver.page_source)
        if dfs:
            df = dfs[0]
            filename = "iron_16mm_data.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"--- Scraping Successful! Data saved to '{filename}' ---")
            return filename
        else:
            print("ERROR: No tables found.")
            return None
    except Exception as e:
        print(f"Scraping Error: {e}")
        return None
    finally:
        driver.quit()

# ==========================================
# PART 2: FORECASTING (16mm)
# ==========================================
def step_2_forecast_16mm(file_name):
    if not file_name or not os.path.exists(file_name):
        return

    print("\n=== STEP 2: PROCESSING 16mm DATA ===")
    
    try:
        # 1. Load Data
        df = pd.read_csv(file_name)
        
        # 2. Transpose
        df_transposed = df.transpose().reset_index()
        df_transposed = df_transposed.rename(columns={'index': 'ds'})
        
        # 3. Find the column with data
        # We assume column 1 has the prices if user selected only 16mm
        target_col = df_transposed.columns[1] 
        
        df_clean = df_transposed[['ds', target_col]].copy()
        df_clean.columns = ['ds', 'y']
        
        # 4. Clean Dates (Arabic Mapping)
        # Using Unicode to avoid Arabic text errors
        arabic_months = {
            '\u064a\u0646\u0627\u064a\u0631': 'Jan', '\u064f\u0628\u0631\u0627\u064a\u0631': 'Feb', '\u0641\u0628\u0631\u0627\u064a\u0631': 'Feb',
            '\u0645\u0627\u0631\u0633': 'Mar', '\u0623\u0628\u0631\u064a\u0644': 'Apr', '\u0627\u0628\u0631\u064a\u0644': 'Apr',
            '\u0645\u0627\u064a\u0648': 'May', '\u064a\u0648\u0646\u064a\u0648': 'Jun', '\u064a\u0648\u0644\u064a\u0648': 'Jul',
            '\u0623\u063a\u0633\u0637\u0633': 'Aug', '\u0627\u063a\u0633\u0637\u0633': 'Aug', '\u0633\u0628\u062a\u0645\u0628\u0631': 'Sep',
            '\u0623\u0643\u062a\u0648\u0628\u0631': 'Oct', '\u0627\u0643\u062a\u0648\u0628\u0631': 'Oct', '\u0646\u0648\u064f\u0645\u0628\u0631': 'Nov',
            '\u0646\u0648\u0641\u0645\u0628\u0631': 'Nov', '\u062f\u064a\u0633\u0645\u0628\u0631': 'Dec'
        }
        
        for ar, en in arabic_months.items():
            df_clean['ds'] = df_clean['ds'].astype(str).str.replace(ar, en, regex=False)

        df_clean['ds'] = pd.to_datetime(df_clean['ds'], errors='coerce')
        
        # 5. Clean Prices
        df_clean['y'] = df_clean['y'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df_clean['y'] = pd.to_numeric(df_clean['y'], errors='coerce')

        # Drop invalid rows
        df_clean = df_clean.dropna().sort_values(by='ds')
        
        print(f"--- Data Rows: {len(df_clean)} ---")
        
        if df_clean.empty:
            print("ERROR: No valid data found.")
            return

        # 6. Prophet Model
        print("--- Training AI Model... ---")
        m = Prophet(daily_seasonality=True)
        m.fit(df_clean)
        
        future = m.make_future_dataframe(periods=365)
        forecast = m.predict(future)
        
        # 7. Plotting
        print("--- Plotting ---")
        m.plot(forecast)
        plt.title('Iron 16mm Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
        
        m.plot_components(forecast)
        plt.show()
        
        # Save Result
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('Iron_16mm_Forecast.csv', index=False)
        print("Done! Saved to 'Iron_16mm_Forecast.csv'")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    csv_file = step_1_scrape_16mm()
    if csv_file:
        step_2_forecast_16mm(csv_file)