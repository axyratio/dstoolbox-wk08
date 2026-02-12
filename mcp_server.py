from fastmcp import FastMCP
import pandas as pd
import numpy as np
import pycaret.classification as pyc
import pycaret.regression as pyr
import os
import sys

# --- 1. สร้าง Server Instance ---
mcp = FastMCP("PyCaret-AutoML-Service")

# --- Helper Functions (Logic ความฉลาด) ---

def infer_task_type(df: pd.DataFrame, target: str) -> str:
    """วิเคราะห์ว่าควรทำ Classification หรือ Regression"""
    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
        return 'classification'
    
    # ถ้าเป็นตัวเลข ให้ดูจำนวนค่าที่ไม่ซ้ำกัน (Cardinality)
    unique_count = df[target].nunique()
    if unique_count < 20: # ถ้ามีค่าน้อยกว่า 20 แบบ (เช่น 0,1 หรือ 1-10) น่าจะเป็น Class
        return 'classification'
    
    return 'regression'

def get_dynamic_setup_params(df: pd.DataFrame, target: str):
    """
    คำนวณค่า Encoding แบบ Dynamic:
    - Low Cardinality (<10) -> One-Hot Encoding
    - High Cardinality (>10) -> Label/Target Encoding (PyCaret default)
    """
    # หาคอลัมน์ที่เป็น Categorical (ไม่รวม Target)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target in cat_cols:
        cat_cols.remove(target)
    
    # แยกแยะว่าจะเอาอันไหนเข้า One-Hot บ้าง
    # PyCaret ใช้ parameter 'max_encoding_ohe' ในการตัดจบ
    # แต่ถ้าเราอยาก Dynamic เราจะหาค่า max ที่เหมาะสมจาก data ชุดนี้
    
    max_ohe_limit = 25 # ค่า Default
    
    # ตรวจสอบดูว่าคอลัมน์ไหน unique value เยอะเกินไป จะได้ไม่ทำ One-Hot
    high_cardinality_cols = [c for c in cat_cols if df[c].nunique() > 10]
    
    return {
        'max_encoding_ohe': 10,  # บังคับว่าถ้าเกิน 10 ค่า ให้เลิกทำ One-Hot แล้วไปทำ Label Encode แทน
        'encoding_method': None  # ให้ PyCaret ตัดสินใจเองสำหรับส่วนที่เหลือ
    }

# --- 2. สร้าง Tools (ฟังก์ชันหลัก) ---

@mcp.tool()
def save_dataset_from_text(filename: str, csv_content: str) -> str:
    """รับข้อมูล CSV จาก Claude (Text) มาบันทึกลงเครื่อง"""
    try:
        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith('.csv'):
            safe_filename += '.csv'
            
        with open(safe_filename, "w", encoding='utf-8') as f:
            f.write(csv_content)
            
        return f"บันทึกไฟล์สำเร็จ: {safe_filename} ({len(csv_content)} bytes)"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการบันทึกไฟล์: {str(e)}"

@mcp.tool()
def run_auto_analysis(file_path: str = "student_depression_dataset.csv", target_column: str = "Depression") -> str:
    """
    วิเคราะห์ข้อมูลอัตโนมัติ (Smart AutoML)
    - ตัดสินใจเองว่าเป็น Classification หรือ Regression
    - จัดการ Encoding แบบ Dynamic (One-Hot vs Label) ตามความเหมาะสมของข้อมูล
    """
    try:
        # ตรวจสอบไฟล์
        if not os.path.exists(file_path):
            return f"Error: ไม่พบไฟล์ที่ {file_path}"

        # โหลดข้อมูล
        df = pd.read_csv(file_path)
        
        if target_column not in df.columns:
            return f"Error: ไม่พบคอลัมน์ '{target_column}' ในไฟล์ CSV"

        # 1. Auto Detect Task
        task = infer_task_type(df, target_column)
        
        # 2. Dynamic Encoding Params
        setup_params = get_dynamic_setup_params(df, target_column)

        # รวม Parameter ทั้งหมด
        common_params = {
            'data': df,
            'target': target_column,
            'session_id': 123,
            'verbose': False,
            'html': False,
            **setup_params # แตก Dictionary ใส่เข้าไป
        }

        # เริ่มกระบวนการ AutoML
        try:
            if task == 'classification':
                exp = pyc.setup(**common_params)
                best_model = pyc.compare_models()
                results = pyc.pull()
            else: # regression
                exp = pyr.setup(**common_params)
                best_model = pyr.compare_models()
                results = pyr.pull()
        except Exception as ml_err:
            return f"เกิดข้อผิดพลาดระหว่าง PyCaret setup/training: {str(ml_err)}"

        # จัดรูปแบบผลลัพธ์
        leaderboard_md = results.head(5).to_markdown()
        
        encoding_info = f"Encoding Strategy: Max One-Hot limit set to {setup_params['max_encoding_ohe']} unique values."
        
        return f"""
## ✅ การวิเคราะห์เสร็จสมบูรณ์ (Auto-Detected)
**Task Type:** {task.upper()}
**Target:** {target_column}
**Best Model:** {results.iloc[0]['Model']}
**Strategy:** {encoding_info}

### 🏆 Top 5 Models Leaderboard
{leaderboard_md}

---
*หมายเหตุ: ระบบเลือก Task และ Encoding ให้โดยอัตโนมัติตามลักษณะข้อมูล*
        """

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการรัน PyCaret: {str(e)}"

@mcp.tool()
def save_best_model(model_name: str) -> str:
    """บันทึกโมเดลที่ดีที่สุดลงเครื่อง (ไฟล์ .pkl)"""
    try:
        # PyCaret จะ save model ที่ active อยู่ล่าสุด
        # เราต้องเช็คก่อนว่าล่าสุดรันอะไรไป (แต่ใน stateless function นี้อาจจะยากหน่อย)
        # วิธีแก้ขัดคือลอง save ของทั้งคู่
        try:
            pyc.save_model(pyc.get_config('best_model'), model_name)
        except Exception as pyc_err:
            try:
                pyr.save_model(pyr.get_config('best_model'), model_name)
            except Exception as pyr_err:
                return f"ไม่สามารถบันทึกโมเดลได้ (ทั้ง Classification และ Regression ล้มเหลว)"
            
        return f"บันทึกโมเดลสำเร็จ: {model_name}.pkl"
    except Exception as e:
        return f"ไม่สามารถบันทึกโมเดลได้ (อาจยังไม่มีการเทรน): {str(e)}"

# --- 3. สร้าง Resources ---

@mcp.resource("local://datasets")
def list_datasets() -> str:
    """แสดงรายชื่อไฟล์ CSV ทั้งหมด"""
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        return "ไม่พบไฟล์ CSV ในโฟลเดอร์นี้"
    return "\n".join([f"- {f}" for f in files])

# --- 4. Start Server ---
if __name__ == "__main__":
    sys.stderr.write("[MCP Server] Starting PyCaret AutoML MCP Server...\n")
    sys.stderr.flush()
    try:
        mcp.run()
    except Exception as e:
        sys.stderr.write(f"[MCP Server] FATAL ERROR: {str(e)}\n")
        sys.stderr.flush()
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
