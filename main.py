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
import time
import uuid
import base64
import random
import textwrap
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
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

# Warm-up các submodule lazy của openai SDK ngay trong main thread lúc khởi động.
# Nếu để các worker thread (ThreadPoolExecutor ở generate_video) cùng truy cập
# openai_client.chat / .audio / .images lần đầu một cách song song, Python import-lock
# sẽ kẹt chéo -> deadlock (_ModuleLock('openai.resources.chat')).
_ = (
    openai_client.chat.completions,
    openai_client.audio.speech,
    openai_client.images,
)

# =========================================================================
# AI VOICE PERSONAS — Nhân vật AI có phong cách riêng
# =========================================================================
VOICE_PERSONAS = [
    {
        "id": "rapper",
        "name": "Rapper Đường Phố",
        "emoji": "🎤",
        "desc": "Nói có vần, slang Gen Z, ngắn gọn bắt tai",
        "voice": "echo",
        "style": "Nói chuyện như rapper: ngắn gọn, có nhịp điệu, hay dùng slang Gen Z (kiểu 'chill', 'vibe', 'flex', 'real talk'), thỉnh thoảng gieo vần tự nhiên. Không dùng từ hoa mỹ."
    },
    {
        "id": "ballad",
        "name": "Ca Sĩ Ballad",
        "emoji": "🎵",
        "desc": "Nhẹ nhàng, cảm xúc, chân thật từ trái tim",
        "voice": "shimmer",
        "style": "Nói chuyện nhẹ nhàng, cảm xúc, hay dùng hình ảnh đẹp và ví von. Giọng điệu chân thật như đang chia sẻ từ trái tim, không vội vàng."
    },
    {
        "id": "mc",
        "name": "MC Năng Động",
        "emoji": "⚡",
        "desc": "Sôi nổi, cuốn hút, đầy năng lượng",
        "voice": "nova",
        "style": "Nói chuyện như MC: năng lượng cao, cuốn hút, hay dùng câu cảm thán, tạo hứng khởi. Nhịp nói nhanh, dứt khoát, tự tin."
    },
    {
        "id": "chidai",
        "name": "Chị Đại Miền Nam",
        "emoji": "👑",
        "desc": "Thẳng thắn, hài hước, chất miền Nam",
        "voice": "alloy",
        "style": "Nói chuyện kiểu chị miền Nam: thẳng thắn, hay xài từ 'nè', 'á', 'hen', 'vậy đó', hài hước tự nhiên, không màu mè. Thỉnh thoảng xổ câu bình dân nghe thân thương."
    },
    {
        "id": "cool",
        "name": "Anh Trai Cool Ngầu",
        "emoji": "😎",
        "desc": "Ít nói, tự tin, câu nào cũng chất",
        "voice": "onyx",
        "style": "Nói ít nhưng câu nào cũng có trọng lượng. Tự tin, không cần giải thích nhiều, hay nói kiểu triết lý ngắn. Không hỏi nhiều, chỉ khẳng định."
    },
    {
        "id": "genz",
        "name": "Cô Nàng Gen Z",
        "emoji": "✨",
        "desc": "Trendy, hài hước, cảm xúc mạnh",
        "voice": "nova",
        "style": "Nói chuyện kiểu Gen Z: hay dùng 'ơi trời', 'thật ra', 'kiểu là', 'không thể tin được', cảm xúc rõ ràng lên xuống. Hài hước nhưng chân thật, hay kể chuyện theo kiểu 'plot twist'."
    },
]

PERSONA_MAP = {p["id"]: p for p in VOICE_PERSONAS}

@app.get("/api/voice-personas")
def get_voice_personas():
    return VOICE_PERSONAS

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


# =========================================================================
# PHẦN 5: DIALOGUE SCRIPT + VOICE — Hội thoại đa nhân vật, ghép audio
# =========================================================================

# Bảng gán giọng cho từng nhân vật (xoay vòng)
VOICE_POOL = ["nova", "onyx", "shimmer", "echo", "alloy"]

class DialogueRequest(BaseModel):
    product_url: str
    persona_a_id: str = "genz"
    persona_b_id: str = "cool"

class DialogueVoiceRequest(BaseModel):
    dialogue: str

@app.post("/api/generate-dialogue")
async def generate_dialogue(req: DialogueRequest):
    if not req.product_url:
        raise HTTPException(status_code=400, detail="Vui lòng nhập link sản phẩm")

    persona_a = PERSONA_MAP.get(req.persona_a_id, VOICE_PERSONAS[0])
    persona_b = PERSONA_MAP.get(req.persona_b_id, VOICE_PERSONAS[1])

    product_info = await extract_product_details(req.product_url)
    product_title = product_info["title"]

    try:
        dialogue = call_ai(
            system="Bạn là biên kịch TikTok chuyên viết kịch bản hội thoại viral cho thị trường Việt Nam. Viết như đời thực, không lộ liễu quảng cáo.",
            user=f"""Sản phẩm: "{product_title}"

Nhân vật A — {persona_a['name']}: {persona_a['style']}
Nhân vật B — {persona_b['name']}: {persona_b['style']}

Viết kịch bản hội thoại TikTok 60 giây. Yêu cầu:
- Mỗi nhân vật nói ĐÚNG phong cách được mô tả, nghe khác biệt rõ ràng
- Bắt đầu bằng tình huống đời thường, KHÔNG nhắc sản phẩm ngay
- Sản phẩm xuất hiện tự nhiên như giải pháp ở giữa video
- Có 1 câu chê nhỏ để tạo độ tin cậy
- Kết thúc nhẹ nhàng, không ép mua

Định dạng mỗi dòng thoại ĐÚNG như sau (không thêm gì khác):
[{persona_a['name']}]: nội dung thoại
[{persona_b['name']}]: nội dung thoại

Viết khoảng 12-16 dòng thoại.""",
            temperature=0.88
        )

        return {
            "status": "success",
            "product_detected": product_title,
            "product_image": product_info.get("image", ""),
            "characters": [
                {"name": persona_a["name"], "emoji": persona_a["emoji"], "desc": persona_a["desc"], "voice": persona_a["voice"], "id": persona_a["id"]},
                {"name": persona_b["name"], "emoji": persona_b["emoji"], "desc": persona_b["desc"], "voice": persona_b["voice"], "id": persona_b["id"]},
            ],
            "dialogue": dialogue
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo kịch bản: {str(e)}")


@app.post("/api/generate-dialogue-voice")
async def generate_dialogue_voice(req: DialogueVoiceRequest):
    if not req.dialogue:
        raise HTTPException(status_code=400, detail="Kịch bản trống")

    # Parse từng dòng thoại: [TÊN|voice]: nội dung hoặc [TÊN]: nội dung
    line_pattern = re.compile(r'^\[(.+?)(?:\|(\w+))?\]:\s*(.+)$')
    lines = []
    char_voice_map = {}
    voice_index = 0

    for raw_line in req.dialogue.strip().split('\n'):
        raw_line = raw_line.strip()
        match = line_pattern.match(raw_line)
        if not match:
            continue
        char_name = match.group(1).strip()
        voice_override = match.group(2)
        text = match.group(3).strip()
        if voice_override and voice_override in [p["voice"] for p in VOICE_PERSONAS]:
            voice = voice_override
        elif char_name not in char_voice_map:
            char_voice_map[char_name] = VOICE_POOL[voice_index % len(VOICE_POOL)]
            voice_index += 1
            voice = char_voice_map[char_name]
        else:
            voice = char_voice_map[char_name]
        lines.append((char_name, text, voice))

    if not lines:
        raise HTTPException(status_code=400, detail="Không parse được dòng thoại nào")

    # Tạo audio từng dòng rồi ghép lại
    try:
        audio_parts = []
        for _, text, voice in lines:
            resp = openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3"
            )
            audio_parts.append(resp.content)

        combined = b''.join(audio_parts)
        return StreamingResponse(
            io.BytesIO(combined),
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'attachment; filename="dialogue_voice.mp3"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi TTS: {str(e)}")


# =========================================================================
# PHẦN 6: VIDEO GENERATOR — Tạo video TikTok 9:16 từ script + voice + ảnh
# =========================================================================
VID_W, VID_H = 720, 1280

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

def _parse_captions(script: str) -> list:
    clean = re.sub(r'\[.*?\]\s*:?', '', script)
    clean = re.sub(r'\n{2,}', '\n', clean).strip()
    sentences = []
    for line in clean.split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'(?<=[.!?])\s+', line)
        for part in parts:
            part = part.strip()
            if len(part) < 5:
                continue
            words = part.split()
            if len(words) > 10:
                for i in range(0, len(words), 9):
                    chunk = ' '.join(words[i:i+9])
                    if chunk:
                        sentences.append(chunk)
            else:
                sentences.append(part)
    return sentences or [script[:200]]

def _build_bg(img_bytes: bytes | None) -> np.ndarray:
    bg = Image.new("RGB", (VID_W, VID_H), (12, 12, 12))
    if img_bytes:
        try:
            prod = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            top_h = VID_H // 2

            # Blurred full-width background
            sq = prod.resize((VID_W, VID_W), Image.LANCZOS)
            blur = sq.filter(ImageFilter.GaussianBlur(35))
            bg.paste(blur.crop((0, 0, VID_W, min(blur.height, top_h))).resize((VID_W, top_h)), (0, 0))

            # Dark overlay
            dark = Image.new("RGB", (VID_W, top_h), (0, 0, 0))
            mask = Image.new("L", (VID_W, top_h), 140)
            bg.paste(dark, (0, 0), mask)

            # Product thumbnail centered in top half
            size = 480
            thumb = prod.resize((size, size), Image.LANCZOS)
            bg.paste(thumb, ((VID_W - size) // 2, (top_h - size) // 2))
        except:
            pass
    return np.array(bg)

def _render_slide(bg_arr: np.ndarray, caption: str, product_name: str) -> np.ndarray:
    img = Image.fromarray(bg_arr.copy())
    cap_y = VID_H // 2 + 20
    cap_h = VID_H - cap_y

    # Semi-transparent caption area
    overlay = Image.new("RGBA", (VID_W, cap_h), (0, 0, 0, 170))
    base = img.convert("RGBA")
    base.paste(overlay, (0, cap_y), overlay)
    img = base.convert("RGB")
    draw = ImageDraw.Draw(img)

    font_name = _get_font(30)
    font_cap = _get_font(54)

    # Product name
    if product_name:
        label = product_name[:44] + "…" if len(product_name) > 44 else product_name
        try:
            draw.text((VID_W // 2, cap_y + 38), label, font=font_name, fill=(180, 180, 180), anchor="mm")
        except TypeError:
            draw.text((10, cap_y + 20), label, font=font_name, fill=(180, 180, 180))

    # Caption text
    wrapped = textwrap.wrap(caption, width=20)
    line_h = 72
    total_h = len(wrapped) * line_h
    y = cap_y + (cap_h - total_h) // 2 + 20

    for line in wrapped:
        try:
            draw.text((VID_W // 2 + 2, y + 2), line, font=font_cap, fill=(0, 0, 0), anchor="mm")
            draw.text((VID_W // 2, y), line, font=font_cap, fill=(255, 255, 255), anchor="mm")
        except TypeError:
            draw.text((40, y), line, font=font_cap, fill=(255, 255, 255))
        y += line_h

    # Progress indicator (thin white line at very bottom)
    draw.rectangle([0, VID_H - 8, VID_W, VID_H], fill=(50, 50, 50))

    return np.array(img)


# =========================================================================
# ANIMATION: anh AI (gpt-image-1) + chuyen dong dien anh (Ken Burns)
# =========================================================================
AW, AH = 480, 854          # do phan giai video animation (9:16, nhe cho free tier — ha xuong de render duoi 100s)
KB_Z = 1.25                # bien du de zoom/pan trong anh
KB_PRESETS = [             # (fx0, fy0, fx1, fy1, zoom_in)
    (0.0, 0.0, 1.0, 1.0, True),
    (1.0, 0.0, 0.0, 1.0, False),
    (0.5, 1.0, 0.5, 0.0, True),
    (0.0, 1.0, 1.0, 0.0, False),
]

def _lerp(a, b, t):
    return a + (b - a) * t

def _fallback_scene(i: int) -> Image.Image:
    """Gradient mau lam canh du phong khi sinh anh AI that bai."""
    pals = [((20,30,60),(120,40,90)), ((10,40,40),(40,120,90)),
            ((60,20,40),(160,90,40)), ((20,20,30),(80,40,120))]
    c1, c2 = pals[i % len(pals)]
    f = (np.arange(AH) / AH)[:, None]
    base = np.zeros((AH, AW, 3), np.uint8)
    for k in range(3):
        base[..., k] = (c1[k] + (c2[k] - c1[k]) * f).astype(np.uint8)
    return Image.fromarray(base)

def _gen_scene_prompts(script: str, product_name: str, n: int) -> list:
    """Dung gpt-4o-mini sinh n prompt anh (tieng Anh) tu noi dung script."""
    excerpt = re.sub(r'\[.*?\]\s*:?', '', script).strip()[:600]
    prompts = []
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"You write vivid cinematic IMAGE prompts (English) for a vertical 9:16 short "
                f"video advertising this product: \"{product_name or 'a product'}\".\n"
                f"Script context:\n{excerpt}\n\n"
                f"Output EXACTLY {n} prompts, one per line, no numbering. Each is one rich visual "
                f"scene (setting, subject, lighting, mood, color) relevant to the product/story, "
                f"energetic and modern for social media. No text or words inside the image."
            )}],
            temperature=0.9,
        )
        lines = [l.strip(" -*\t0123456789.") for l in r.choices[0].message.content.splitlines()]
        prompts = [l for l in lines if len(l) > 8][:n]
    except Exception:
        prompts = []
    while len(prompts) < n:
        prompts.append(f"Cinematic vibrant lifestyle scene featuring {product_name or 'a trendy product'}, "
                       f"dynamic lighting, modern, energetic, vertical composition, ultra detailed")
    return prompts

def _gen_scene_images(prompts: list) -> list:
    """Sinh anh canh bang gpt-image-1 (song song). Tra ve list PIL hoac None."""
    def gen(pr):
        try:
            r = openai_client.images.generate(
                model="gpt-image-1", prompt=pr, size="1024x1536", quality="low", n=1
            )
            return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=4) as ex:
        return list(ex.map(gen, prompts))

def _cover_base(img: Image.Image) -> Image.Image:
    """Resize-cover anh ve khung lon hon frame (de con bien Ken Burns)."""
    BW, BH = int(AW * KB_Z), int(AH * KB_Z)
    iw, ih = img.size
    scale = max(BW / iw, BH / ih)
    img = img.resize((int(iw * scale) + 1, int(ih * scale) + 1), Image.BILINEAR)
    iw, ih = img.size
    left = (iw - BW) // 2; top = (ih - BH) // 2
    return img.crop((left, top, left + BW, top + BH))

def _kb_frame(base: Image.Image, p: float, preset) -> Image.Image:
    """Cat 1 cua so zoom/pan tu base theo tien do p -> resize ve (AW, AH)."""
    BW, BH = base.size
    fx0, fy0, fx1, fy1, zin = preset
    vf = _lerp(1.0, 0.85, p) if zin else _lerp(0.85, 1.0, p)
    vis_w = vf * BW; vis_h = vf * BH
    mx = BW - vis_w; my = BH - vis_h
    x = mx * _lerp(fx0, fx1, p); y = my * _lerp(fy0, fy1, p)
    return base.crop((int(x), int(y), int(x + vis_w), int(y + vis_h))).resize((AW, AH), Image.BILINEAR)

def _build_anim_make_frame(scenes, captions, duration, product_name):
    """Tra ve make_frame: Ken Burns + crossfade giua canh + caption fade-in."""
    bases = [_cover_base(s) for s in scenes]
    n_s = len(bases); n_c = max(1, len(captions))
    sdur = duration / n_s; cdur = duration / n_c
    td = min(0.5, sdur * 0.4)

    scrim_h = 380
    ys = (np.arange(scrim_h) / scrim_h) ** 1.4
    scrim_arr = np.zeros((scrim_h, AW, 4), np.uint8)
    scrim_arr[..., 3] = (210 * ys).astype(np.uint8)[:, None]
    scrim = Image.fromarray(scrim_arr, "RGBA")
    font_cap = _get_font(40)
    font_pn = _get_font(22)

    def scene_at(t):
        si = min(int(t / sdur), n_s - 1)
        p = (t - si * sdur) / sdur
        f = _kb_frame(bases[si], min(p, 1.0), KB_PRESETS[si % len(KB_PRESETS)])
        s_end = (si + 1) * sdur
        if si < n_s - 1 and t > s_end - td:
            a = (t - (s_end - td)) / td
            nf = _kb_frame(bases[si + 1], 0.0, KB_PRESETS[(si + 1) % len(KB_PRESETS)])
            f = Image.blend(f, nf, a)
        return f

    def make_frame(t):
        base = scene_at(t).convert("RGBA")
        base.alpha_composite(scrim, (0, AH - scrim_h))
        draw = ImageDraw.Draw(base)
        ci = min(int(t / cdur), n_c - 1)
        cp = (t - ci * cdur) / cdur
        alpha = min(1.0, cp / 0.2)
        wrapped = textwrap.wrap(captions[ci], width=18)
        line_h = 52
        y = AH - 70 - len(wrapped) * line_h
        tc = (255, 255, 255, int(255 * alpha)); sc = (0, 0, 0, int(180 * alpha))
        for ln in wrapped:
            try:
                draw.text((AW // 2 + 2, y + 2), ln, font=font_cap, fill=sc, anchor="mm")
                draw.text((AW // 2, y), ln, font=font_cap, fill=tc, anchor="mm")
            except TypeError:
                draw.text((30, y), ln, font=font_cap, fill=tc)
            y += line_h
        if product_name:
            label = product_name[:40] + "…" if len(product_name) > 40 else product_name
            try:
                draw.text((AW // 2, AH - scrim_h + 34), label, font=font_pn,
                          fill=(220, 220, 220, 230), anchor="mm")
            except TypeError:
                pass
        return np.array(base.convert("RGB"))

    return make_frame


class VideoRequest(BaseModel):
    script: str
    voice: str = "nova"            # giữ để tương thích cũ; video giờ luôn hội thoại đa giọng
    product_image_url: str = ""
    product_name: str = ""
    persona_a_id: str = "genz"     # nhân vật A (xem VOICE_PERSONAS)
    persona_b_id: str = "cool"     # nhân vật B


def _scriptify_to_dialogue(script: str, product_name: str,
                           persona_a: dict, persona_b: dict) -> str:
    """Chuyen kich ban doc 1 giong -> hoi thoai 2 nhan vat noi chuyen qua lai.
    Tra ve text dinh dang moi dong [Ten|voice]: loi  (de _synth_video_audio ghep da giong).
    """
    # Bo cac marker dao dien [BOI CANH QUAY], [HOOK]... chi giu noi dung
    excerpt = re.sub(r'\[.*?\]\s*:?', '', script).strip()[:1500]
    raw = call_ai(
        system="Bạn là biên kịch TikTok chuyên viết hội thoại tự nhiên, viral cho thị trường Việt Nam. Chỉ trả về các dòng thoại, không giải thích.",
        user=f"""Sản phẩm: "{product_name or 'sản phẩm'}"

Kịch bản gốc (dạng độc thoại) cần chuyển thể:
\"\"\"
{excerpt}
\"\"\"

Hãy CHUYỂN thành đoạn hội thoại TikTok giữa 2 nhân vật nói chuyện QUA LẠI với nhau,
giữ nguyên thông điệp và mạch câu chuyện (vấn đề → giải pháp → lời kêu gọi mua).

Nhân vật A — {persona_a['name']}: {persona_a['style']}
Nhân vật B — {persona_b['name']}: {persona_b['style']}

Yêu cầu:
- Hai người ĐỐI THOẠI qua lại (hỏi - đáp, phản ứng), KHÔNG phải mỗi người đọc một đoạn dài.
- Mỗi nhân vật nói đúng phong cách riêng, nghe khác biệt rõ.
- Nửa đầu nêu vấn đề/tình huống, chưa khoe sản phẩm. Giữ 1 câu chê nhỏ cho chân thật.
- Kết bằng lời kêu gọi mua nhẹ nhàng.
- Khoảng 10-16 lượt thoại, mỗi lượt 1-2 câu ngắn gọn.

Định dạng MỖI dòng ĐÚNG như sau (không markdown, không số thứ tự, không thêm gì khác):
[{persona_a['name']}|{persona_a['voice']}]: lời thoại
[{persona_b['name']}|{persona_b['voice']}]: lời thoại""",
        temperature=0.85,
    )
    return raw.strip()


def _parse_dialogue_captions(dialogue: str) -> list:
    """Moi luot thoai -> caption (chia nho neu dai), kem ten nguoi noi o dau luot."""
    line_pattern = re.compile(r'^\[(.+?)(?:\|(\w+))?\]:\s*(.+)$')
    caps = []
    for raw in dialogue.strip().split('\n'):
        m = line_pattern.match(raw.strip())
        if not m:
            continue
        name = m.group(1).strip()
        text = m.group(3).strip()
        words = text.split()
        if len(words) > 12:
            for i in range(0, len(words), 11):
                chunk = ' '.join(words[i:i + 11])
                caps.append(f"{name}: {chunk}" if i == 0 else chunk)
        else:
            caps.append(f"{name}: {text}")
    return caps or _parse_captions(dialogue)


def _synth_video_audio(script: str, default_voice: str) -> bytes:
    """Sinh audio cho video.
    - Neu script la hoi thoai (nhieu dong dang [Ten|voice]: ... hoac [Ten]: ...)
      thi doc tung dong bang giong rieng cua moi nhan vat roi ghep lai (da giong).
    - Nguoc lai (script don) thi doc bang 1 giong `default_voice`.
    """
    line_pattern = re.compile(r'^\[(.+?)(?:\|(\w+))?\]:\s*(.+)$')
    valid_voices = [p["voice"] for p in VOICE_PERSONAS]
    parsed = []
    char_voice_map = {}
    voice_index = 0

    for raw in script.strip().split('\n'):
        m = line_pattern.match(raw.strip())
        if not m:
            continue
        char_name = m.group(1).strip()
        voice_override = m.group(2)
        text = m.group(3).strip()
        if voice_override and voice_override in valid_voices:
            voice = voice_override
        elif char_name not in char_voice_map:
            char_voice_map[char_name] = VOICE_POOL[voice_index % len(VOICE_POOL)]
            voice_index += 1
            voice = char_voice_map[char_name]
        else:
            voice = char_voice_map[char_name]
        parsed.append((text, voice))

    # Hoi thoai: >=2 dong co tag nhan vat -> ghep da giong.
    # Goi TTS song song (giu nguyen thu tu) de giam thoi gian cho.
    if len(parsed) >= 2:
        def _tts(item):
            text, voice = item
            resp = openai_client.audio.speech.create(
                model="tts-1", voice=voice, input=text, response_format="mp3"
            )
            return resp.content
        with ThreadPoolExecutor(max_workers=4) as ex:
            parts = list(ex.map(_tts, parsed))
        return b''.join(parts)

    # Script don -> 1 giong
    clean = re.sub(r'\[.*?\]\s*:?', '', script)
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Script rỗng sau khi làm sạch")
    resp = openai_client.audio.speech.create(
        model="tts-1", voice=default_voice, input=clean, response_format="mp3"
    )
    return resp.content


def _build_video_sync(req: VideoRequest) -> bytes:
    """Toan bo pipeline tao video, chay DONG BO trong 1 thread nen (worker job).
    Tra ve bytes MP4. Nem Exception (kem message) neu loi -> job luu vao status error.

    Video luon o dang HOI THOAI: chuyen kich ban -> doan thoai 2 nhan vat, doc da giong.
    """
    persona_a = PERSONA_MAP.get(req.persona_a_id, VOICE_PERSONAS[0])
    persona_b = PERSONA_MAP.get(req.persona_b_id, VOICE_PERSONAS[1])
    # Neu chon trung 1 nhan vat -> ep B khac A de van co 2 giong
    if persona_b["id"] == persona_a["id"]:
        persona_b = next((p for p in VOICE_PERSONAS if p["id"] != persona_a["id"]), persona_b)

    # Buoc 1: chuyen kich ban doc 1 giong -> hoi thoai 2 nhan vat
    dialogue = _scriptify_to_dialogue(req.script, req.product_name, persona_a, persona_b)

    captions = _parse_dialogue_captions(dialogue)
    # So canh AI = ceil(captions/4), gioi han 2..3 de kiem soat chi phi
    n_scene = max(2, min(3, -(-len(captions) // 4)))

    # Tai anh san pham that (neu co) -> dung lam canh cuoi (product reveal)
    img_bytes = None
    if req.product_image_url:
        try:
            with httpx.Client(timeout=8) as c:
                r = c.get(req.product_image_url)
                if r.status_code == 200:
                    img_bytes = r.content
        except Exception:
            pass

    # Chay song song: tong hop audio hoi thoai (da giong) + sinh anh canh bang AI.
    # Scene prompts dung kich ban GOC (con marker [BOI CANH QUAY]...) de co goi y hinh anh.
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_audio = ex.submit(_synth_video_audio, dialogue, persona_a["voice"])
        f_scenes = ex.submit(
            lambda: _gen_scene_images(_gen_scene_prompts(req.script, req.product_name, n_scene))
        )
        audio_bytes = f_audio.result()
        scenes = f_scenes.result()

    # Thay canh sinh loi bang gradient du phong -> luon du canh
    scenes = [s if s is not None else _fallback_scene(i) for i, s in enumerate(scenes)]
    if img_bytes:
        try:
            scenes.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        except Exception:
            pass
    if not scenes:
        scenes = [_fallback_scene(0)]

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = os.path.join(tmp, "audio.mp3")
        video_path = os.path.join(tmp, "video.mp4")

        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        from moviepy.editor import VideoClip, AudioFileClip

        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        # Ken Burns (zoom/pan) tren anh AI + crossfade + caption fade-in.
        # make_frame chi giu 1 khung trong RAM -> nhe cho free tier.
        make_frame = _build_anim_make_frame(scenes, captions, duration, req.product_name)

        video = VideoClip(make_frame, duration=duration)
        video = video.set_audio(audio_clip)
        video.write_videofile(
            video_path, fps=20, codec="libx264", audio_codec="aac",
            preset="ultrafast", threads=2,
            temp_audiofile=os.path.join(tmp, "tmp_audio.m4a"),
            remove_temp=True, logger=None
        )

        with open(video_path, "rb") as f:
            mp4 = f.read()

        video.close()
        audio_clip.close()

    return mp4


# =========================================================================
# JOB BAT DONG BO — POST tra ngay job_id, frontend poll trang thai roi tai ket qua.
# Tranh gioi han ~100s edge timeout cua Cloudflare/Render cho video dai.
# Luu trong RAM (free tier chi 1 worker). Job tu het han sau JOB_TTL.
# =========================================================================
_jobs: dict = {}
_jobs_lock = threading.Lock()
JOB_TTL = 1800  # 30 phut

def _cleanup_jobs():
    now = time.time()
    with _jobs_lock:
        for k in [k for k, v in _jobs.items() if now - v["ts"] > JOB_TTL]:
            _jobs.pop(k, None)

def _set_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _jobs[job_id]["ts"] = time.time()

def _run_video_job(job_id: str, req: VideoRequest):
    _set_job(job_id, status="processing")
    try:
        mp4 = _build_video_sync(req)
        _set_job(job_id, status="done", mp4=mp4)
    except Exception as e:
        _set_job(job_id, status="error", error=str(e) or "Lỗi không xác định")


@app.post("/api/generate-video")
def generate_video(req: VideoRequest):
    """Khoi tao job tao video, tra ngay job_id (khong cho render xong)."""
    if not req.script:
        raise HTTPException(status_code=400, detail="Script trống")
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "mp4": None, "error": None, "ts": time.time()}
    threading.Thread(target=_run_video_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/video-status/{job_id}")
def video_status(job_id: str):
    """Hoi trang thai job: pending | processing | done | error."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job không tồn tại hoặc đã hết hạn")
        return {"status": job["status"], "error": job["error"]}


@app.get("/api/video-result/{job_id}")
def video_result(job_id: str):
    """Tai MP4 khi job xong. Lay xong thi xoa khoi RAM."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job không tồn tại hoặc đã hết hạn")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Lỗi tạo video")
    if job["status"] != "done" or not job["mp4"]:
        raise HTTPException(status_code=409, detail="Video chưa sẵn sàng")
    mp4 = job["mp4"]
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return StreamingResponse(
        io.BytesIO(mp4), media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="tiktok_video.mp4"'}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
