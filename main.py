from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import httpx
from playwright.async_api import async_playwright
import os
import io
import re
import json
import random
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

app = FastAPI(title="TikTok AI Script Factory")

@app.get("/")
def root():
    return {"status": "ok", "service": "TikTok AI Script Factory"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================================
# SHARED: Scrape product info
# =========================================================================
async def extract_product_details(url: str) -> dict:
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
                og_title = soup.find("meta", property="og:title")
                title = og_title["content"].strip() if og_title and og_title.get("content") else ""
                if "Mua và Bán" in title or title == "Shopee Việt Nam":
                    title = ""
                og_image = soup.find("meta", property="og:image")
                image_url = og_image["content"].strip() if og_image and og_image.get("content") else ""
                if title:
                    return {"title": title, "image": image_url}
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
                img_el = await page.query_selector("meta[itemprop='image']")
                if img_el:
                    image_url = await img_el.get_attribute("content")
            except:
                pass
            await browser.close()
            is_trash = "Mua và Bán" in raw_title or raw_title.strip() == "Shopee Việt Nam" or not raw_title
            if not is_trash:
                title = raw_title.replace("| Shopee Việt Nam", "").replace("Shopee Việt Nam", "").strip()
                return {"title": title, "image": image_url}
    except:
        pass

    return {"title": "Sản phẩm Shopee", "image": ""}


def call_ai(system: str, user: str, temperature: float = 0.8) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


# =========================================================================
# PHẦN 1: TRENDING HOOKS — AI-generated, không hardcode
# =========================================================================
_hooks_cache: list = []

def _generate_hooks_from_ai() -> list:
    raw = call_ai(
        system="Bạn là chuyên gia phân tích trend TikTok Việt Nam. Chỉ trả về JSON, không giải thích.",
        user="""Tạo 8 câu Hook TikTok đang viral cho thị trường Việt Nam (2 câu mỗi danh mục: Trend, Drama, Dễ làm, Bán hàng).
Trả về JSON array đúng format sau, không có markdown:
[
  {"category": "Trend", "niche": "Thời trang", "text": "Câu hook ở đây", "views": "1.2M"},
  ...
]"""
    )
    try:
        cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        hooks = json.loads(cleaned)
        for i, h in enumerate(hooks):
            h["id"] = i + 1
        return hooks
    except:
        return []

@app.get("/api/trending-hooks")
def get_trending_hooks(category: str = "Tất cả"):
    global _hooks_cache
    if not _hooks_cache:
        _hooks_cache = _generate_hooks_from_ai()
    if category == "Tất cả":
        return _hooks_cache
    return [h for h in _hooks_cache if h.get("category") == category]

@app.post("/api/ai-update-trends")
def ai_update_trends():
    global _hooks_cache
    new_hooks = _generate_hooks_from_ai()
    if not new_hooks:
        raise HTTPException(status_code=500, detail="AI không tạo được hooks mới")
    _hooks_cache = new_hooks
    return {"status": "success", "message": "Đã cập nhật xu hướng mới!", "count": len(new_hooks)}


# =========================================================================
# PHẦN 2: SCRIPT ĐƠN — 1 kịch bản theo tone
# =========================================================================
class ScriptRequest(BaseModel):
    product_url: str
    tone: str

@app.post("/api/generate-script")
async def generate_script(req: ScriptRequest):
    if not req.product_url:
        raise HTTPException(status_code=400, detail="Vui lòng nhập link sản phẩm")

    product_info = await extract_product_details(req.product_url)

    try:
        script = call_ai(
            system="Bạn là Content Creator TikTok tài năng. Bạn bán hàng 'như không bán'. Lồng ghép sản phẩm vào tình huống đời thường một cách chân thật nhất.",
            user=f"""Sản phẩm: "{product_info['title']}"
Tone giọng: {req.tone}

Viết kịch bản video 45 giây:
1. Nửa đầu KHÔNG nhắc tên sản phẩm. Bắt đầu bằng vấn đề/câu chuyện.
2. Giữa video: đưa sản phẩm ra như "vị cứu tinh". Khen 1 điểm, chê 1 điểm nhỏ cho chân thật.
3. Cuối: gợi ý mua hàng nhẹ nhàng.

Định dạng bắt buộc:
[BỐI CẢNH QUAY]:
[HOOK - 3s đầu]:
[STORYTELLING - 15s]:
[GIẢI PHÁP TỰ NHIÊN - 15s]:
[CTA TINH TẾ - 5s]:""",
            temperature=0.75
        )
        return {
            "status": "success",
            "product_detected": product_info["title"],
            "product_image": product_info.get("image", ""),
            "script": script
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi AI: {str(e)}")


# =========================================================================
# PHẦN 3: SCRIPT 3 PERSONA — AI tự chọn persona phù hợp sản phẩm
# =========================================================================
class PersonaScriptRequest(BaseModel):
    product_url: str

@app.post("/api/generate-persona-scripts")
async def generate_persona_scripts(req: PersonaScriptRequest):
    if not req.product_url:
        raise HTTPException(status_code=400, detail="Vui lòng nhập link sản phẩm")

    product_info = await extract_product_details(req.product_url)
    product_title = product_info["title"]

    # Bước 1: AI phân tích sản phẩm và đề xuất 3 persona phù hợp nhất
    try:
        personas_raw = call_ai(
            system="Bạn là chuyên gia marketing TikTok Việt Nam. Chỉ trả về JSON, không giải thích.",
            user=f"""Sản phẩm: "{product_title}"

Phân tích sản phẩm và đề xuất 3 nhân vật (persona) người dùng TikTok Việt Nam phù hợp NHẤT để quảng bá sản phẩm này.
Mỗi persona phải có câu chuyện cá nhân liên quan trực tiếp đến sản phẩm.

Trả về JSON array đúng format, không markdown:
[
  {{
    "id": 1,
    "emoji": "emoji phù hợp",
    "name": "Tên nhân vật ngắn gọn",
    "desc": "Mô tả 1 dòng (tuổi · đặc điểm · pain point)",
    "story_angle": "Góc kể chuyện cho nhân vật này với sản phẩm trên"
  }},
  ...3 items...
]"""
        )
        cleaned = personas_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        personas = json.loads(cleaned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích persona: {str(e)}")

    # Bước 2: AI viết kịch bản cho từng persona
    try:
        persona_list_str = "\n".join([
            f"PERSONA {p['id']}: {p['name']} — {p['desc']}\nGóc kể chuyện: {p['story_angle']}"
            for p in personas
        ])

        scripts_raw = call_ai(
            system="Bạn là chuyên gia viết kịch bản TikTok bán hàng affiliate Việt Nam. Viết tự nhiên, chân thật, không lộ liễu quảng cáo.",
            user=f"""Sản phẩm: "{product_title}"

{persona_list_str}

Viết 3 kịch bản TikTok 60 giây, mỗi kịch bản cho 1 persona. Dùng đúng marker phân tách.

===PERSONA_1===
[0-3s] HOOK:
[3-15s] VẤN ĐỀ:
[15-40s] GIẢI PHÁP:
[40-55s] DEMO:
[55-60s] CTA:

===PERSONA_2===
[0-3s] HOOK:
[3-15s] VẤN ĐỀ:
[15-40s] GIẢI PHÁP:
[40-55s] DEMO:
[55-60s] CTA:

===PERSONA_3===
[0-3s] HOOK:
[3-15s] VẤN ĐỀ:
[15-40s] GIẢI PHÁP:
[40-55s] DEMO:
[55-60s] CTA:

Quy tắc: Nửa đầu KHÔNG nhắc tên sản phẩm. Khen 1 điểm, chê 1 điểm nhỏ cho chân thật.""",
            temperature=0.8
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi viết kịch bản: {str(e)}")

    # Parse scripts theo marker
    scripts = []
    for i, persona in enumerate(personas):
        marker = f"===PERSONA_{i+1}==="
        next_marker = f"===PERSONA_{i+2}===" if i < len(personas) - 1 else None
        start = scripts_raw.find(marker)
        end = scripts_raw.find(next_marker) if next_marker else len(scripts_raw)
        block = scripts_raw[start + len(marker):end].strip() if start != -1 else ""
        scripts.append({**persona, "script": block})

    return {
        "status": "success",
        "product_detected": product_title,
        "product_image": product_info.get("image", ""),
        "scripts": scripts
    }


# =========================================================================
# PHẦN 4: VOICE SCRIPT — TTS từ kịch bản
# =========================================================================
class VoiceRequest(BaseModel):
    script: str
    voice: str = "nova"  # nova, alloy, onyx, echo, fable, shimmer

@app.post("/api/generate-voice")
async def generate_voice(req: VoiceRequest):
    if not req.script:
        raise HTTPException(status_code=400, detail="Kịch bản trống")

    # Xóa các stage direction như [HOOK - 3s đầu]:, [BỐI CẢNH QUAY]: ...
    clean = re.sub(r'\[.*?\]\s*:?', '', req.script)
    # Xóa dòng trống thừa
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()

    if not clean:
        raise HTTPException(status_code=400, detail="Không có nội dung để đọc")

    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice=req.voice,
            input=clean,
            response_format="mp3"
        )
        audio_bytes = response.content
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="voice_script.mp3"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi TTS: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
