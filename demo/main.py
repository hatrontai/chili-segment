from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import base64
from PIL import Image

app = FastAPI()

# Cho phép Frontend truy cập vào API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def mock_model_predict(input_image):
    """Giả lập hàm xử lý model của bạn"""
    # Xử lý ảnh (ví dụ: xoay ảnh)
    output_img = input_image.rotate(180)
    
    # Tạo DataFrame kết quả
    df = pd.DataFrame({
        "Feature": ["Accuracy", "Latency", "Object Count", "Score"],
        "Value": [0.98, "45ms", 12, "High"]
    })
    return output_img, df

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Đọc ảnh từ request
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # 2. Gọi model xử lý
    res_image, res_df = mock_model_predict(image)

    # 3. Chuyển ảnh kết quả sang Base64 string
    buffered = io.BytesIO()
    res_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # 4. Trả về JSON
    return {
        "result_image": f"data:image/jpeg;base64,{img_str}",
        "table_data": res_df.to_dict(orient="records")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)