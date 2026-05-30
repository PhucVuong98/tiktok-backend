from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
from playwright.async_api import async_playwright
import os
import re
import urllib.parse
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import random

load_dotenv()

app = FastAPI(title="TikTok AI Script Factory - Full Version")

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ScriptRequest(BaseModel):
    product_url: str
    tone: str

# =========================================================================
# PHẦN 1: TÍNH NĂNG TRENDING HOOKS & AI CẬP NHẬT TREND
# =========================================================================
TRENDING_HOOKS_DB = [
    {"id": 1, "category": "Trend", "niche": "Đa ngành", "text": "Top 3 sản phẩm đang làm mưa làm gió tuần này mà bạn chưa biết.", "views": "1.2M"},
    {"id": 2, "category": "Trend", "niche": "Thời trang", "text": "Đu trend muộn còn hơn không, set đồ này đang quá cháy!", "views": "900K"},
    {"id": 3, "category": "Drama", "niche": "Review", "text": "Sự thật mất lòng về món đồ này mà không shop nào dám nói cho bạn.", "views": "2.1M"},
    {"id": 4, "category": "Drama", "niche": "Đời sống", "text": "Bóc phốt cách làm mà mọi người vẫn tin sái cổ bấy lâu nay.", "views": "1.5M"},
    {"id": 5, "category": "Dễ làm", "niche": "Mẹo vặt", "text": "Chỉ mất đúng 30 giây để giải quyết triệt để vấn đề này.", "views": "850K"},
    {"id": 6, "category": "Bán hàng", "niche": "Làm đẹp", "text": "Bà nào đang tốn tiền oan thì bơi hết vào video này ngay.", "views": "1.8M"},
    {"id": 7, "category": "Bán hàng", "niche": "Đa ngành", "text": "Mình đã định không mua đâu, cho đến khi thấy tính năng thứ 2 của em nó...", "views": "1.1M"}
]

@app.get("/api/trending-hooks")
def get_trending_hooks(category: str = "Tất cả"):
    if category == "Tất cả":
        return TRENDING_HOOKS_DB
    return [h for h in TRENDING_HOOKS_DB if h["category"] == category]

@app.post("/api/ai-update-trends")
def ai_update_trends():
    system_prompt = "Bạn là một AI phân tích dữ liệu TikTok Việt Nam."
    user_prompt = """
    Hãy đóng vai chuyên gia bắt trend, sáng tạo ra 4 câu Hook (Mở đầu video) cực kỳ cuốn hút, đánh trúng tâm lý người xem ngay lúc này. 
    Yêu cầu trả về đúng 4 câu thuộc 4 danh mục: Trend, Drama, Dễ làm, Bán hàng.
    Trả về ĐÚNG định dạng có chứa dấu phẩy phân cách như sau (không nói gì thêm):
    Trend|[Ngách]|Nội dung câu hook
    Drama|[Ngách]|Nội dung câu hook
    Dễ làm|[Ngách]|Nội dung câu hook
    Bán hàng|[Ngách]|Nội dung câu hook
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8
        )
        raw_text = response.choices[0].message.content.strip().split("\n")
        new_hooks = []
        for line in raw_text:
            parts = line.split("|")
            if len(parts) >= 3:
                new_hooks.append({
                    "id": random.randint(100, 999),
                    "category": parts[0].strip(),
                    "niche": parts[1].strip(),
                    "text": parts[2].strip(),
                    "views": f"{random.randint(100, 999)}K"
                })
        global TRENDING_HOOKS_DB
        TRENDING_HOOKS_DB = new_hooks + TRENDING_HOOKS_DB
        return {"status": "success", "message": "Đã cập nhật xu hướng mới!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# PHẦN 2: TÍNH NĂNG CÀO DỮ LIỆU SHOPEE & TẠO KỊCH BẢN TỰ NHIÊN
# =========================================================================
async def extract_product_details(url: str) -> dict:
    print(f"\n[LOG] Đang xử lý link: {url}")
    current_url = url.strip()
    
    if "shopee.vn" not in current_url or "shp.ee" in current_url:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
                response = await client.get(current_url, headers=headers)
                current_url = str(response.url)
        except:
            pass

    try:
        bot_headers = {
            "User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_patched.html)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9"
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            res = await client.get(current_url, headers=bot_headers)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, "html.parser")
                og_title_tag = soup.find("meta", property="og:title")
                title = ""
                if og_title_tag and og_title_tag.get("content"):
                    title = og_title_tag["content"].strip()
                    if "Mua và Bán" in title or "Shopee Việt Nam" == title:
                        title = "" 
                
                og_image_tag = soup.find("meta", property="og:image")
                image_url = og_image_tag["content"].strip() if og_image_tag and og_image_tag.get("content") else ""

                if title:
                    return {"title": title, "image": image_url, "description": "Lấy thành công."}
    except:
        pass

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0",
                locale="vi-VN"
            )
            page = await context.new_page()
            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
            
            await page.goto(current_url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(4000) 
            
            raw_title = await page.title()
            image_url = ""
            try:
                img_element = await page.query_selector("meta[itemprop='image']")
                if img_element:
                    image_url = await img_element.get_attribute("content")
            except:
                pass

            is_trash_title = "Mua và Bán" in raw_title or "Shopee Việt Nam" == raw_title.strip() or not raw_title
            if not is_trash_title:
                playwright_title = raw_title.replace("| Shopee Việt Nam", "").replace("Shopee Việt Nam", "").strip()
                await browser.close()
                return {"title": playwright_title, "image": image_url, "description": "Lấy bằng Playwright."}
                
            await browser.close()
    except:
        pass

    return {"title": "Sản phẩm Shopee", "image": "", "description": "Không thể lấy thông tin"}


@app.post("/api/generate-script")
async def generate_script(req: ScriptRequest):
    if not req.product_url:
        raise HTTPException(status_code=400, detail="Vui lòng nhập link sản phẩm")
    
    product_info = await extract_product_details(req.product_url)
    
    system_prompt = "Bạn là một Content Creator TikTok tài năng. Bạn bán hàng 'như không bán'. Lồng ghép sản phẩm vào tình huống đời thường một cách chân thật nhất."
    
    user_prompt = f"""
Sản phẩm cần lồng ghép: "{product_info['title']}"
Tone giọng: {req.tone}

Hãy viết kịch bản video ngắn (45 giây):
1. Nửa đầu video tuyệt đối KHÔNG NHẮC ĐẾN TÊN SẢN PHẨM. Bắt đầu bằng một vấn đề/câu chuyện.
2. Giữa video: Đưa sản phẩm ra như một "vị cứu tinh". Khen 1 điểm, chê 1 điểm nhỏ cho chân thật.
3. Cuối video: Gợi ý mua hàng cực kỳ nhẹ nhàng.

Định dạng bắt buộc:
[BỐI CẢNH QUAY]: 
[HOOK - 3s đầu]: 
[STORYTELLING - 15s]: 
[GIẢI PHÁP TỰ NHIÊN - 15s]: 
[CTA TINH TẾ - 5s]: 
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.75 
        )
        script_generated = response.choices[0].message.content
        
        return {
            "status": "success",
            "product_detected": product_info['title'],
            "product_image": product_info.get('image', ''),
            "script": script_generated
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi AI Engine: {str(e)}")

# Khởi chạy server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    