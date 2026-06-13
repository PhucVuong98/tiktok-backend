from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, FileResponse
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
import math
import random
import textwrap
import tempfile
import threading
import queue as _queue
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from auth_credits import require_uid, get_credits, reserve_credit, refund_credit, admin_project_id

load_dotenv()

# Gia credit cho 1 lan tao video. Doi qua env neu sau nay can.
VIDEO_CREDIT_COST = int(os.getenv("VIDEO_CREDIT_COST", "1"))

# MOCK_RENDER=1: tao video GIA bang PIL/moviepy, KHONG goi OpenAI (LLM/TTS/image).
# Dung de test luong queue/credit/poll/download ma khong ton phi. Mac dinh TAT.
MOCK_RENDER = os.getenv("MOCK_RENDER", "").lower() in ("1", "true", "yes")

app = FastAPI(title="TikTok AI Script Factory")

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TikTok AI Script Factory",
        "mock_render": MOCK_RENDER,
        "firebase_project": admin_project_id(),   # đối chiếu với project frontend (vincent-ai-tiktok)
    }

@app.get("/api/me")
def me(uid: str = Depends(require_uid)):
    """Thong tin tai khoan dang nhap: so credit con lai (frontend hien thi + chan nut)."""
    return {"uid": uid, "credits": get_credits(uid)}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# timeout=90s + max_retries=1: khong de bat ky call OpenAI nao treo lau (truoc day
# 1 call sinh anh treo ~10 phut khien ca job dung -> "Quá thời gian tạo video").
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=90.0, max_retries=1)

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
def ai_update_trends(uid: str = Depends(require_uid)):
    global _hooks_cache
    new_hooks = _generate_hooks_from_ai()
    if not new_hooks:
        raise HTTPException(status_code=500, detail="AI không tạo được hooks mới")
    _hooks_cache = new_hooks
    return {"status": "success", "message": "Đã cập nhật xu hướng mới!", "count": len(new_hooks)}


# =========================================================================
# PHẦN 1B: GIÁ VÀNG — cập nhật giá vàng trong nước (PNJ) + thế giới (goldprice.org)
# Endpoint CÔNG KHAI, $0 (chỉ HTTP, KHÔNG gọi OpenAI). Tra ve them `script_seed`
# (text mo ta gia da format san, KHONG LLM) de frontend dua thang vao /api/generate-video.
# =========================================================================
_GOLD_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_GOLD_CACHE_TTL = 60          # giay: tranh hammer nguon khi nhieu user refresh
_gold_cache: dict = {"data": None, "ts": 0.0}

def _fetch_world_gold() -> dict | None:
    """Gia vang the gioi (XAU, USD/ounce). Ghep 2 nguon mien phi, khong key:
    - gold-api.com: gia spot chinh xac (on dinh, nhung khong co % thay doi).
    - coingecko PAX Gold (PAXG, bam sat gia vang): bo sung % thay doi 24h.
    Tra ve None neu ca 2 deu loi."""
    price = pct = None
    try:
        with httpx.Client(timeout=10, headers=_GOLD_UA) as c:
            r = c.get("https://api.gold-api.com/price/XAU")
        price = round(float(r.json()["price"]), 2)
    except Exception:
        pass
    try:
        with httpx.Client(timeout=10, headers=_GOLD_UA) as c:
            r = c.get("https://api.coingecko.com/api/v3/simple/price",
                      params={"ids": "pax-gold", "vs_currencies": "usd",
                              "include_24hr_change": "true"})
        pg = r.json()["pax-gold"]
        pct = round(float(pg["usd_24h_change"]), 2)
        if price is None:
            price = round(float(pg["usd"]), 2)
    except Exception:
        pass
    if price is None:
        return None
    prev = change = None
    if pct is not None:
        prev = round(price / (1 + pct / 100), 2)
        change = round(price - prev, 2)
    return {"price_usd": price, "change": change, "pct_change": pct,
            "prev_close": prev, "unit": "USD/ounce"}

def _fetch_domestic_gold(limit: int = 6) -> tuple[list, str]:
    """Gia vang trong nuoc tu PNJ edge API. Tra ve (list, updateDate). [] neu loi.
    PNJ tra giaban/giamua theo NGHIN DONG / chi -> nhan 1000 ra VND."""
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get("https://edge-api.pnj.io/ecom-frontend/v1/get-gold-price", headers=_GOLD_UA)
        d = r.json()
        out = []
        for it in d.get("data", [])[:limit]:
            try:
                out.append({
                    "name": str(it.get("tensp", "")).strip(),
                    "buy": int(round(float(it["giamua"]) * 1000)),
                    "sell": int(round(float(it["giaban"]) * 1000)),
                })
            except Exception:
                continue
        return out, str(d.get("updateDate", ""))
    except Exception:
        return [], ""

def _parse_gold_num(s: str | None) -> int | None:
    """'14,700' (nghin dong/chi) -> 14700000 VND. '-' / rong -> None."""
    if not s:
        return None
    digits = s.strip().replace(",", "").replace(".", "")
    if not digits.isdigit():
        return None
    return int(digits) * 1000

def _fetch_doji_gold(limit: int = 6) -> tuple[list, str]:
    """Gia vang DOJI tu XML feed cong khai. Tra ve (rows, updated). [] neu loi.
    Bo qua hang chi co gia mua (nguyen lieu) cho bang gon."""
    try:
        with httpx.Client(timeout=10, headers=_GOLD_UA, follow_redirects=True) as c:
            r = c.get("https://update.giavang.doji.vn/banggia/doji_92411/get")
        root = ET.fromstring(r.text)
        dt = root.find(".//DateTime")
        upd = dt.text.strip() if dt is not None and dt.text else ""
        rows = []
        for row in root.findall(".//Row"):
            name = (row.get("Name") or "").strip()
            sell = _parse_gold_num(row.get("Sell"))
            buy = _parse_gold_num(row.get("Buy"))
            if name and sell:           # bo hang khong co gia ban (vd nguyen lieu 18k)
                rows.append({"name": name, "buy": buy, "sell": sell})
            if len(rows) >= limit:
                break
        return rows, upd
    except Exception:
        return [], ""

def _vnd_trieu(v: int) -> str:
    """VND -> chuoi 'X,Y trieu' (dau phay thap phan kieu VN)."""
    return f"{v / 1_000_000:.1f}".replace(".", ",") + " triệu"

def _gold_script_seed(world: dict | None, domestic: list) -> str:
    """Ghep text mo ta gia vang (KHONG LLM) -> dua vao pipeline video de scriptify."""
    parts = ["Cập nhật giá vàng hôm nay!"]
    if world:
        px = f'{world["price_usd"]:,.0f}'.replace(",", ".")
        if world.get("pct_change") is not None:
            d = "tăng" if world["pct_change"] >= 0 else "giảm"
            parts.append(f'Vàng thế giới đang ở mức {px} đô la Mỹ mỗi ounce, '
                         f'{d} {abs(world["pct_change"])} phần trăm so với phiên trước.')
        else:
            parts.append(f'Vàng thế giới đang ở mức {px} đô la Mỹ mỗi ounce.')
    if domestic:
        g = domestic[0]
        if g.get("buy"):
            parts.append(f'Trong nước, {g["name"]} mua vào {_vnd_trieu(g["buy"])}, '
                         f'bán ra {_vnd_trieu(g["sell"])} mỗi chỉ.')
        else:
            parts.append(f'Trong nước, {g["name"]} bán ra {_vnd_trieu(g["sell"])} mỗi chỉ.')
        if len(domestic) > 1:
            g2 = domestic[1]
            parts.append(f'{g2["name"]} bán ra {_vnd_trieu(g2["sell"])} mỗi chỉ.')
    parts.append("Vàng đang là kênh đầu tư được nhiều người quan tâm, "
                 "anh em nhớ theo dõi để cập nhật giá mới nhất nhé!")
    return " ".join(parts)

@app.get("/api/gold-prices")
def gold_prices():
    """Gia vang trong nuoc + the gioi (CONG KHAI, $0). Cache 60s. `script_seed` san de tao video."""
    now = time.time()
    if _gold_cache["data"] and now - _gold_cache["ts"] < _GOLD_CACHE_TTL:
        return _gold_cache["data"]

    world = _fetch_world_gold()
    pnj_rows, pnj_upd = _fetch_domestic_gold()
    doji_rows, doji_upd = _fetch_doji_gold()

    # domestic = danh sach nguon, moi nguon co bang gia rieng (chi giu nguon lay duoc)
    domestic = []
    if pnj_rows:
        domestic.append({"source": "PNJ", "updated": pnj_upd, "rows": pnj_rows})
    if doji_rows:
        domestic.append({"source": "DOJI", "updated": doji_upd, "rows": doji_rows})

    if world is None and not domestic:
        # Tat ca nguon fail -> dung cache cu neu co, khong thi 503
        if _gold_cache["data"]:
            return {**_gold_cache["data"], "stale": True}
        raise HTTPException(status_code=503, detail="Không lấy được dữ liệu giá vàng, thử lại sau nhé.")

    # script_seed dung bang gia cua nguon dau tien lay duoc (PNJ uu tien)
    primary_rows = domestic[0]["rows"] if domestic else []
    data = {
        "world": world,
        "domestic": domestic,
        "script_seed": _gold_script_seed(world, primary_rows),
        "stale": False,
    }
    _gold_cache["data"] = data
    _gold_cache["ts"] = now
    return data


# =========================================================================
# PHẦN 2: SCRIPT ĐƠN — 1 kịch bản theo tone
# =========================================================================
class ScriptRequest(BaseModel):
    product_url: str
    tone: str

@app.post("/api/generate-script")
async def generate_script(req: ScriptRequest, uid: str = Depends(require_uid)):
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
async def generate_persona_scripts(req: PersonaScriptRequest, uid: str = Depends(require_uid)):
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
async def generate_voice(req: VoiceRequest, uid: str = Depends(require_uid)):
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
async def generate_dialogue(req: DialogueRequest, uid: str = Depends(require_uid)):
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
async def generate_dialogue_voice(req: DialogueVoiceRequest, uid: str = Depends(require_uid)):
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

def _gen_scene_prompts(script: str, product_name: str, n: int, product_focus: bool = False) -> list:
    """Dung gpt-4o-mini sinh n prompt anh (tieng Anh) tu noi dung script.
    product_focus=True (mode review 1 nhan vat): moi canh DAT SAN PHAM lam trung tam."""
    excerpt = re.sub(r'\[.*?\]\s*:?', '', script).strip()[:600]
    focus = (
        (f"This is a PRODUCT REVIEW. The product \"{product_name or 'the product'}\" MUST be the "
         f"clear hero/centerpiece of EVERY scene (close-ups, product in use, lifestyle with the "
         f"product). ") if product_focus else ""
    )
    prompts = []
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": (
                f"You write vivid cinematic IMAGE prompts (English) for a vertical 9:16 short "
                f"video advertising this product: \"{product_name or 'a product'}\".\n"
                f"{focus}"
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

# Chan dung AI cho moi persona (mo ta tieng Anh cho gpt-image-1 ra anh dep hon).
PERSONA_PORTRAIT = {
    "rapper": "a young Vietnamese male street rapper wearing a cap and hoodie, confident cool look",
    "ballad": "a gentle young Vietnamese female ballad singer with a soft warm smile",
    "mc": "an energetic young Vietnamese male TV host in a smart outfit, bright friendly smile",
    "chidai": "a confident mature Vietnamese woman, warm strong expression, Southern Vietnam vibe",
    "cool": "a cool stylish young Vietnamese man with a calm confident expression",
    "genz": "a trendy cheerful young Vietnamese Gen Z girl, cute and fashionable",
}
_portrait_cache: dict = {}
_portrait_lock = threading.Lock()

def _gen_persona_portrait(persona: dict):
    """Sinh 1 anh chan dung 1024x1024 (mieng ngam) cho persona. Tra ve PIL hoac None."""
    desc = PERSONA_PORTRAIT.get(persona["id"], f"a friendly Vietnamese person ({persona['name']})")
    try:
        r = openai_client.images.generate(
            model="gpt-image-1",
            prompt=(f"Front-facing portrait headshot of {desc}. Head and shoulders, centered, "
                    f"looking straight at camera, mouth closed, neutral friendly expression, "
                    f"clean solid-color studio background, soft even lighting, vibrant modern "
                    f"social-media style, highly detailed. No text, no watermark."),
            size="1024x1024", quality="low", n=1,
        )
        return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
    except Exception:
        return None

def _gen_open_mouth(base1024: Image.Image):
    """Tu anh chan dung (mieng ngam) tao ban HA MIENG bang images.edit + mask vung mieng.
    Giu nguyen guong mat (chi sua vung mieng). Tra ve PIL 1024 hoac None."""
    try:
        d = 1024
        img = base1024.convert("RGB").resize((d, d), Image.LANCZOS)
        # mask RGBA: vung TRONG SUOT (alpha=0) la vung se duoc chinh -> dat o mieng
        mask = Image.new("RGBA", (d, d), (0, 0, 0, 255))
        cx, cy = d // 2, int(d * 0.66)
        rw, rh = int(d * 0.16), int(d * 0.11)
        ImageDraw.Draw(mask).ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=(0, 0, 0, 0))
        bi = io.BytesIO(); img.save(bi, "PNG"); bi.seek(0); bi.name = "image.png"
        bm = io.BytesIO(); mask.save(bm, "PNG"); bm.seek(0); bm.name = "mask.png"
        r = openai_client.images.edit(
            model="gpt-image-1", image=bi, mask=bm, size="1024x1024", n=1,
            prompt=("Same person, same face and lighting, but with the mouth OPEN as if "
                    "talking mid-sentence, natural open mouth slightly showing teeth."),
        )
        return Image.open(io.BytesIO(base64.b64decode(r.data[0].b64_json))).convert("RGB")
    except Exception:
        return None

def _get_persona_faces(persona: dict):
    """Tra ve (anh_ngam_512, anh_ha_512_or_None), cache theo id de khong sinh lai."""
    pid = persona["id"]
    with _portrait_lock:
        if pid in _portrait_cache:
            return _portrait_cache[pid]
    base = _gen_persona_portrait(persona)
    if base is None:
        return (None, None)
    opened = _gen_open_mouth(base)
    faces = (base.resize((512, 512), Image.LANCZOS),
             opened.resize((512, 512), Image.LANCZOS) if opened is not None else None)
    with _portrait_lock:
        _portrait_cache[pid] = faces
    return faces

def _circle_portrait(img: Image.Image, d: int) -> Image.Image:
    """Cat anh thanh hinh tron duong kinh d (RGBA, vien mem)."""
    p = img.convert("RGB").resize((d, d), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, d - 1, d - 1], fill=255)
    p.putalpha(mask)
    return p

# =========================================================================
# ANIMATION NANG CAP (thuan PIL/numpy -> $0, KHONG goi them API):
#  1) Chuyen canh da dang: crossfade / slide-push / zoom-punch / whip-pan
#  2) Caption karaoke: hien tung chu theo loi noi (chu dang toi = vang + nhich len)
#  3) Sticker sparkle: ngoi sao lap lanh bay len khi caption co tu "hype"
# =========================================================================
SCENE_TRANSITIONS = ["crossfade", "slide", "zoom", "whip"]

def _zoom_center(img: Image.Image, z: float) -> Image.Image:
    """Phong to anh quanh tam roi crop ve dung kich thuoc cu (cho zoom-punch)."""
    if z <= 1.0:
        return img
    w, h = img.size
    big = img.resize((int(w * z), int(h * z)), Image.BILINEAR)
    bw, bh = big.size
    left = (bw - w) // 2; top = (bh - h) // 2
    return big.crop((left, top, left + w, top + h))

def _transition(f_cur: Image.Image, f_next: Image.Image, a: float, kind: str) -> Image.Image:
    """Tron 2 khung canh (AW x AH) theo tien do a (0->1) voi kieu chuyen canh `kind`."""
    a = max(0.0, min(1.0, a))
    if kind == "slide":          # day ngang: canh cu truot trai, canh moi vao tu phai
        dx = int(a * AW)
        out = Image.new("RGB", (AW, AH))
        out.paste(f_cur, (-dx, 0)); out.paste(f_next, (AW - dx, 0))
        return out
    if kind == "zoom":           # canh moi "dam" vao: zoom 1.18 -> 1.0 + blend
        nf = _zoom_center(f_next, _lerp(1.18, 1.0, a))
        return Image.blend(f_cur, nf, a)
    if kind == "whip":           # quat ngang mo: 2 khung mo dan giua chang + truot
        r = math.sin(a * math.pi) * 7.0
        fc = f_cur.filter(ImageFilter.GaussianBlur(r)) if r > 0.3 else f_cur
        fn = f_next.filter(ImageFilter.GaussianBlur(r)) if r > 0.3 else f_next
        dx = int(a * AW)
        out = Image.new("RGB", (AW, AH))
        out.paste(fc, (-dx, 0)); out.paste(fn, (AW - dx, 0))
        return out
    return Image.blend(f_cur, f_next, a)   # crossfade (mac dinh)

# --- Caption karaoke ----------------------------------------------------
def _draw_karaoke(draw, text, cp, font, y0, line_h, width=18, max_lines=3):
    """Ve caption hien dan tung chu. cp = tien do trong caption (0..1).
    Chu da qua = trang, chu dang toi = vang + nhich len, chu chua toi = mo nhe."""
    lines = textwrap.wrap(text, width=width)[:max_lines]
    words = []                       # (line_idx, word) phang, gi = chi so toan cuc
    for li, ln in enumerate(lines):
        for w in ln.split():
            words.append((li, w))
    if not words:
        return
    total = len(words)
    reveal = cp / 0.88 * total       # hoan tat o 88% caption -> chu cuoi kip sang
    active = int(reveal)
    frac = reveal - active
    sp = draw.textlength(" ", font=font)
    for li, ln in enumerate(lines):
        line_words = [(gi, w) for gi, (lj, w) in enumerate(words) if lj == li]
        widths = [draw.textlength(w, font=font) for _, w in line_words]
        total_w = sum(widths) + sp * max(0, len(line_words) - 1)
        x = (AW - total_w) / 2
        y = y0 + li * line_h
        for (gi, w), wdt in zip(line_words, widths):
            if gi < active:
                col = (255, 255, 255, 255); dy = 0
            elif gi == active:
                pop = frac * frac * (3 - 2 * frac)            # ease-in-out
                col = (255, 222, 64, 255); dy = -int(7 * (1 - pop))
            else:
                col = (255, 255, 255, 70); dy = 0
            cx = x + wdt / 2
            try:
                draw.text((cx + 2, y + 2 + dy), w, font=font, fill=(0, 0, 0, 150), anchor="mm")
                draw.text((cx, y + dy), w, font=font, fill=col, anchor="mm")
            except TypeError:
                draw.text((x, y + dy), w, font=font, fill=col)
            x += wdt + sp

# --- Sticker sparkle ----------------------------------------------------
_HYPE_SET = {
    "xịn", "xin", "hot", "đỉnh", "dinh", "wow", "trend", "cháy", "chay", "mê", "me",
    "yêu", "yeu", "thích", "thich", "tuyệt", "tuyet", "ngon", "chất", "chat", "hời", "hoi",
    "sale", "giảm", "giam", "freeship", "sốc", "soc", "real", "vibe", "must", "deal",
}

def _has_hype(text: str) -> bool:
    return bool(set(re.findall(r"\w+", text.lower())) & _HYPE_SET)

def _sparkles_for(ci: int, n: int = 5):
    """Sinh n sparkle co vi tri/pha co dinh (seed theo ci) -> deterministic giua cac frame."""
    rng = random.Random(ci * 1000 + 7)
    return [{
        "x": rng.randint(int(AW * 0.12), int(AW * 0.88)),
        "y": rng.randint(int(AH * 0.46), int(AH * 0.70)),
        "delay": rng.uniform(0.0, 0.50),     # le pha xuat hien
        "life": rng.uniform(0.45, 0.75),     # song bao lau (ti le voi caption)
        "r": rng.randint(10, 20),
        "rise": rng.randint(70, 140),        # bay len bao nhieu px
        "warm": rng.random() < 0.5,          # vang am hay trang
    } for _ in range(n)]

def _star_points(cx, cy, r, rs, rot=0.0, n=4):
    pts = []
    for i in range(n * 2):
        rad = r if i % 2 == 0 else rs
        ang = rot + math.pi * i / n
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts

def _draw_sparkles(draw, sparkles, cp):
    """Ve sparkle cho caption hien tai. cp = tien do trong caption (0..1)."""
    for s in sparkles:
        lp = (cp - s["delay"]) / s["life"]    # tien do song rieng cua sparkle
        if lp < 0.0 or lp > 1.0:
            continue
        if lp < 0.25:           af = lp / 0.25
        elif lp > 0.65:         af = max(0.0, (1.0 - lp) / 0.35)
        else:                   af = 1.0
        alpha = int(235 * af)
        if alpha <= 4:
            continue
        twinkle = 0.6 + 0.4 * abs(math.sin(cp * 12 + s["x"]))   # lap lanh
        rr = s["r"] * twinkle
        cx = s["x"]; cy = s["y"] - s["rise"] * lp               # bay len
        col = (255, 226, 130, alpha) if s["warm"] else (255, 255, 255, alpha)
        draw.polygon(_star_points(cx, cy, rr, rr * 0.34, cp * 2.0 + s["x"]), fill=col)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=col)


CH_R = 150                      # ban kinh chan dung nhan vat lon

def _build_anim_make_frame(scenes, captions, duration, product_name,
                           characters=None, caption_speakers=None):
    """Tra ve make_frame: Ken Burns + nhan vat lon dang noi (map may mieng + nhun) + caption.
    characters: list dict {name, closed, open} (PIL 512 hoac None).
    caption_speakers: ten nguoi noi moi caption."""
    bases = [_cover_base(s) for s in scenes]
    n_s = len(bases); n_c = max(1, len(captions))
    sdur = duration / n_s; cdur = duration / n_c
    td = min(0.5, sdur * 0.4)

    scrim_h = 360
    ys = (np.arange(scrim_h) / scrim_h) ** 1.4
    scrim_arr = np.zeros((scrim_h, AW, 4), np.uint8)
    scrim_arr[..., 3] = (210 * ys).astype(np.uint8)[:, None]
    scrim = Image.fromarray(scrim_arr, "RGBA")
    font_cap = _get_font(40)
    font_pn = _get_font(22)
    font_nm = _get_font(24)     # ten nhan vat

    char_cx = AW // 2
    char_cy = 430
    D = 2 * CH_R

    # Sticker sparkle: caption nao co tu "hype" thi co san bo sparkle (seed co dinh)
    cap_sparkles = [_sparkles_for(i) if _has_hype(c) else None
                    for i, c in enumerate(captions)]

    # Chuan bi san anh tron ngam/ha mieng cho moi nhan vat (khong resize lai moi frame)
    ch_circ = {}
    for ch in (characters or []):
        cl = _circle_portrait(ch["closed"], D) if ch.get("closed") is not None else None
        op = _circle_portrait(ch["open"], D) if ch.get("open") is not None else None
        ch_circ[ch["name"]] = (cl, op)

    def _draw_character(base, draw, name, cstart, t):
        pair = ch_circ.get(name)
        if not pair or pair[0] is None:
            return False
        closed, opened = pair
        # map may mieng: doi ngam/ha moi ~0.1s khi dang noi
        use_open = opened is not None and (int(t / 0.1) % 2 == 1)
        img = opened if use_open else closed
        # "nay" nhe khi vao luot moi (scale 0.9 -> 1.0) + nhun len xuong nhe
        pop = min(1.0, (t - cstart) / 0.22)
        scale = 0.9 + 0.1 * (pop * pop * (3 - 2 * pop))
        bob = int(np.sin(t * 9.0) * 4)
        d = max(8, int(D * scale))
        im = img.resize((d, d), Image.BILINEAR)
        x = char_cx - d // 2; y = char_cy - d // 2 + bob
        # vong sang phia sau cho noi bat
        draw.ellipse([char_cx - d // 2 - 5, y - 5, char_cx + d // 2 + 5, y + d + 5],
                     outline=(255, 255, 255, 230), width=4)
        base.alpha_composite(im, (x, y))
        try:
            draw.text((char_cx, char_cy + CH_R + 26), name[:18], font=font_nm,
                      fill=(255, 255, 255, 255), anchor="mm")
        except TypeError:
            pass
        return True

    def scene_at(t):
        si = min(int(t / sdur), n_s - 1)
        p = (t - si * sdur) / sdur
        f = _kb_frame(bases[si], min(p, 1.0), KB_PRESETS[si % len(KB_PRESETS)])
        s_end = (si + 1) * sdur
        if si < n_s - 1 and t > s_end - td:
            a = (t - (s_end - td)) / td
            nf = _kb_frame(bases[si + 1], 0.0, KB_PRESETS[(si + 1) % len(KB_PRESETS)])
            # Chuyen canh luan phien: crossfade / slide / zoom-punch / whip-pan
            f = _transition(f, nf, a, SCENE_TRANSITIONS[(si + 1) % len(SCENE_TRANSITIONS)])
        return f

    def make_frame(t):
        base = scene_at(t).convert("RGBA")
        base.alpha_composite(scrim, (0, AH - scrim_h))
        draw = ImageDraw.Draw(base)
        ci = min(int(t / cdur), n_c - 1)
        cp = (t - ci * cdur) / cdur

        # Nhan vat lon dang noi
        drew = False
        if characters:
            spk = caption_speakers[ci] if caption_speakers and ci < len(caption_speakers) else None
            if spk is not None:
                drew = _draw_character(base, draw, spk, ci * cdur, t)
        if not drew and product_name:
            label = product_name[:40] + "…" if len(product_name) > 40 else product_name
            try:
                draw.text((AW // 2, AH - scrim_h + 30), label, font=font_pn,
                          fill=(220, 220, 220, 230), anchor="mm")
            except TypeError:
                pass

        # Sticker sparkle (ve duoi caption de khong de chu) khi caption co tu "hype"
        if ci < len(cap_sparkles) and cap_sparkles[ci] is not None:
            _draw_sparkles(draw, cap_sparkles[ci], cp)

        # Caption karaoke: hien tung chu theo loi noi (toi da 3 dong)
        line_h = 50
        n_lines = len(textwrap.wrap(captions[ci], width=18)[:3])
        y0 = AH - 55 - n_lines * line_h
        _draw_karaoke(draw, captions[ci], cp, font_cap, y0, line_h, width=18, max_lines=3)

        return np.array(base.convert("RGB"))

    return make_frame


class VideoRequest(BaseModel):
    script: str
    voice: str = "nova"            # giữ để tương thích cũ
    product_image_url: str = ""
    product_name: str = ""
    mode: str = "dialogue"         # "dialogue" = 2 nhân vật hội thoại; "single" = 1 nhân vật review
    persona_a_id: str = "genz"     # nhân vật A / hoặc người review (mode single)
    persona_b_id: str = "cool"     # nhân vật B (chỉ mode dialogue)


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
- NGẮN GỌN cho video 25-30 giây: khoảng 8-11 lượt thoại, mỗi lượt 1 câu ngắn (toàn bài ~90-105 từ). Cô đọng, bỏ câu thừa.

Định dạng MỖI dòng ĐÚNG như sau (không markdown, không số thứ tự, không thêm gì khác):
[{persona_a['name']}|{persona_a['voice']}]: lời thoại
[{persona_b['name']}|{persona_b['voice']}]: lời thoại""",
        temperature=0.85,
    )
    return raw.strip()


def _scriptify_to_review(script: str, product_name: str, persona: dict) -> str:
    """Viet lai kich ban thanh loi REVIEW 1 nguoi noi, tu nhien, theo phong cach persona.
    Tra ve text thuong (cau noi lien mach), KHONG marker/[..] de TTS + caption sach."""
    excerpt = re.sub(r'\[.*?\]\s*:?', '', script).strip()[:1500]
    raw = call_ai(
        system=("Bạn là một TikTok reviewer Việt Nam nói chuyện tự nhiên, duyên dáng. "
                "Chỉ trả về lời nói, không tiêu đề, không markdown, không ký hiệu trong ngoặc vuông."),
        user=f"""Sản phẩm: "{product_name or 'sản phẩm'}"

Kịch bản gốc cần viết lại:
\"\"\"
{excerpt}
\"\"\"

Hãy VIẾT LẠI thành lời tự review sản phẩm của MỘT người nói (first-person), nghe thật tự nhiên,
như đang quay TikTok review thật, đúng phong cách nhân vật:

Nhân vật — {persona['name']}: {persona['style']}

Yêu cầu:
- Một người nói liền mạch (không hội thoại, không tên người nói, không stage direction).
- Giữ mạch: mở đầu nêu vấn đề/tình huống → giới thiệu sản phẩm như giải pháp → 1 câu chê nhỏ cho chân thật → kết bằng lời kêu gọi mua nhẹ nhàng.
- Nói đúng phong cách nhân vật, dùng từ ngữ tự nhiên của họ.
- NGẮN GỌN cho video 25-30 giây (khoảng 90-105 từ). Cô đọng, mỗi câu một ý, bỏ câu thừa.
- CHỈ trả về phần lời nói thuần, không ký hiệu [], không gạch đầu dòng.""",
        temperature=0.85,
    )
    # Don sach phong khi model lo chen marker
    return re.sub(r'\[.*?\]\s*:?', '', raw).strip()


def _parse_dialogue_captions(dialogue: str) -> list:
    """Moi luot thoai -> 1 hoac nhieu (ten_nguoi_noi, caption). Chia nho luot dai."""
    line_pattern = re.compile(r'^\[(.+?)(?:\|(\w+))?\]:\s*(.+)$')
    caps = []
    for raw in dialogue.strip().split('\n'):
        m = line_pattern.match(raw.strip())
        if not m:
            continue
        name = m.group(1).strip()
        text = m.group(3).strip()
        words = text.split()
        if len(words) > 11:
            for i in range(0, len(words), 10):
                caps.append((name, ' '.join(words[i:i + 10])))
        else:
            caps.append((name, text))
    if not caps:
        return [(None, c) for c in _parse_captions(dialogue)]
    return caps


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


def _synth_single_voice(script: str, voice: str) -> bytes:
    """Doc THANG script bang 1 giong (mode review 1 nhan vat).
    Xoa stage direction [BOI CANH QUAY]/[HOOK]... roi doc lien mach -> khong hieu nham
    cac marker la nhan vat nhu _synth_video_audio."""
    clean = re.sub(r'\[.*?\]\s*:?', '', script)
    clean = re.sub(r'\n{3,}', '\n\n', clean).strip()
    if not clean:
        raise HTTPException(status_code=400, detail="Script rỗng sau khi làm sạch")
    resp = openai_client.audio.speech.create(
        model="tts-1", voice=voice, input=clean, response_format="mp3"
    )
    return resp.content


def _build_video_sync(req: VideoRequest) -> bytes:
    """Toan bo pipeline tao video, chay DONG BO trong 1 thread nen (worker job).
    Tra ve bytes MP4. Nem Exception (kem message) neu loi -> job luu vao status error.

    2 mode:
    - "single": 1 nhan vat tu review -> 1 giong doc thang script, KHONG hoat hinh nhan vat,
      background bam sat san pham.
    - "dialogue" (mac dinh): chuyen kich ban -> hoi thoai 2 nhan vat, doc da giong + nhan vat noi.
    """
    single = (req.mode == "single")

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

    if single:
        # ===== MODE 1 NHAN VAT REVIEW =====
        persona = PERSONA_MAP.get(req.persona_a_id, VOICE_PERSONAS[0])
        # Viet lai script thanh loi review tu nhien theo phong cach persona
        review = _scriptify_to_review(req.script, req.product_name, persona)
        captions = _parse_captions(review)
        caption_speakers = None
        characters = None
        n_scene = max(2, min(3, -(-len(captions) // 4)))
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_audio = ex.submit(_synth_single_voice, review, persona["voice"])
            f_scenes = ex.submit(
                lambda: _gen_scene_images(
                    _gen_scene_prompts(req.script, req.product_name, n_scene, product_focus=True))
            )
            audio_bytes = f_audio.result()
            scenes = f_scenes.result()
    else:
        # ===== MODE HOI THOAI 2 NHAN VAT (mac dinh) =====
        persona_a = PERSONA_MAP.get(req.persona_a_id, VOICE_PERSONAS[0])
        persona_b = PERSONA_MAP.get(req.persona_b_id, VOICE_PERSONAS[1])
        if persona_b["id"] == persona_a["id"]:   # trung -> ep B khac de van co 2 giong
            persona_b = next((p for p in VOICE_PERSONAS if p["id"] != persona_a["id"]), persona_b)

        dialogue = _scriptify_to_dialogue(req.script, req.product_name, persona_a, persona_b)
        caps = _parse_dialogue_captions(dialogue)
        captions = [t for _, t in caps]
        caption_speakers = [n for n, _ in caps]
        n_scene = max(2, min(3, -(-len(captions) // 4)))

        # audio hoi thoai (da giong) + anh canh AI + chan dung 2 nhan vat (cache theo id).
        with ThreadPoolExecutor(max_workers=4) as ex:
            f_audio = ex.submit(_synth_video_audio, dialogue, persona_a["voice"])
            f_scenes = ex.submit(
                lambda: _gen_scene_images(_gen_scene_prompts(req.script, req.product_name, n_scene))
            )
            f_fa = ex.submit(_get_persona_faces, persona_a)
            f_fb = ex.submit(_get_persona_faces, persona_b)
            audio_bytes = f_audio.result()
            scenes = f_scenes.result()
            faces_a = f_fa.result()
            faces_b = f_fb.result()

        characters = [
            {"name": persona_a["name"], "closed": faces_a[0], "open": faces_a[1]},
            {"name": persona_b["name"], "closed": faces_b[0], "open": faces_b[1]},
        ]

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
        make_frame = _build_anim_make_frame(
            scenes, captions, duration, req.product_name,
            characters=characters, caption_speakers=caption_speakers,
        )

        video = VideoClip(make_frame, duration=duration)
        video = video.set_audio(audio_clip)
        video.write_videofile(
            video_path, fps=15, codec="libx264", audio_codec="aac",
            preset="ultrafast", threads=2,
            temp_audiofile=os.path.join(tmp, "tmp_audio.m4a"),
            remove_temp=True, logger=None
        )

        with open(video_path, "rb") as f:
            mp4 = f.read()

        video.close()
        audio_clip.close()

    return mp4


def _build_mock_video(req: VideoRequest) -> bytes:
    """MOCK: tao MP4 hop le nhung KHONG goi OpenAI (khong LLM/TTS/anh AI).
    Dung cac canh gradient du phong (_fallback_scene) + caption tu chinh script,
    khong audio -> render rat nhanh, $0. Bat bang env MOCK_RENDER=1.
    Caption gan tien to [MOCK] de khong nham la video that."""
    captions = [f"[MOCK] {c}" for c in (_parse_captions(req.script)[:6] or ["Mock video"])]
    scenes = [_fallback_scene(i) for i in range(3)]
    duration = max(3.0, min(8.0, len(captions) * 1.2))
    make_frame = _build_anim_make_frame(
        scenes, captions, duration, req.product_name or "MOCK PRODUCT",
        characters=None, caption_speakers=None,
    )
    with tempfile.TemporaryDirectory() as tmp:
        video_path = os.path.join(tmp, "video.mp4")
        from moviepy.editor import VideoClip
        video = VideoClip(make_frame, duration=duration)
        video.write_videofile(
            video_path, fps=12, codec="libx264", audio=False,
            preset="ultrafast", threads=2, logger=None,
        )
        with open(video_path, "rb") as f:
            mp4 = f.read()
        video.close()
    return mp4


# =========================================================================
# JOB BAT DONG BO + HANG DOI CO GIOI HAN (bounded queue)
# POST tra ngay job_id -> frontend poll status -> tai result. Ne edge timeout ~100s.
#
# Khac ban cu (de chay SaaS that, khong sap khi dong user tren free tier):
#  - Render TUAN TU qua hang doi (MAX_WORKERS, free tier = 1) thay vi de-thread vo
#    han -> tranh OOM khi nhieu nguoi bam cung luc.
#  - Tu choi (503) khi hang doi qua MAX_QUEUE -> co backpressure, khong nhan vo han.
#  - MP4 ket qua luu xuong DISK (khong giu bytes trong RAM) -> nhe RAM 512MB.
#  - Tra them `position` (vi tri xep hang) cho frontend hien thi.
# =========================================================================
_RESULT_DIR = os.path.join(tempfile.gettempdir(), "tiktok_jobs")
os.makedirs(_RESULT_DIR, exist_ok=True)

JOB_TTL = 1800                                          # 30 phut: don job + file cu
MAX_WORKERS = int(os.getenv("VIDEO_WORKERS", "1"))      # so video render dong thoi (free tier nen = 1)
MAX_QUEUE = int(os.getenv("VIDEO_QUEUE_MAX", "20"))     # so job cho toi da truoc khi tu choi

_jobs: dict = {}                  # job_id -> {status, error, path, ts}
_jobs_lock = threading.Lock()
_job_q: "_queue.Queue[str]" = _queue.Queue()
_job_reqs: dict = {}              # job_id -> VideoRequest (tach khoi _jobs)


def _cleanup_jobs():
    now = time.time()
    with _jobs_lock:
        stale = [k for k, v in _jobs.items() if now - v["ts"] > JOB_TTL]
        for k in stale:
            v = _jobs.pop(k, None)
            _job_reqs.pop(k, None)
            if v and v.get("path") and os.path.exists(v["path"]):
                try:
                    os.remove(v["path"])
                except OSError:
                    pass


def _set_job(job_id: str, **fields):
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)
            _jobs[job_id]["ts"] = time.time()


def _video_worker():
    """Worker chay mai mai: lay job tu hang doi va render TUAN TU.
    So worker = MAX_WORKERS gioi han so video render dong thoi -> bao ve RAM/CPU free tier."""
    while True:
        job_id = _job_q.get()
        try:
            req = _job_reqs.get(job_id)
            if req is None:               # job da bi don (het han) truoc khi toi luot
                continue
            _set_job(job_id, status="processing")
            mp4 = _build_mock_video(req) if MOCK_RENDER else _build_video_sync(req)
            path = os.path.join(_RESULT_DIR, f"{job_id}.mp4")
            with open(path, "wb") as f:
                f.write(mp4)
            _set_job(job_id, status="done", path=path)
        except Exception as e:
            _set_job(job_id, status="error", error=str(e) or "Lỗi không xác định")
            # Render that bai -> hoan lai credit da tru cho user (cong bang, ko mat tien oan).
            with _jobs_lock:
                job = _jobs.get(job_id) or {}
            if job.get("uid") and job.get("cost"):
                refund_credit(job["uid"], job["cost"])
        finally:
            _job_reqs.pop(job_id, None)
            _job_q.task_done()


# Khoi dong pool worker co dinh ngay khi import module (1 worker tren free tier).
for _ in range(max(1, MAX_WORKERS)):
    threading.Thread(target=_video_worker, daemon=True).start()


@app.post("/api/generate-video")
def generate_video(req: VideoRequest, uid: str = Depends(require_uid)):
    """Khoi tao job tao video, tra ngay job_id (khong cho render xong).
    TRU credit truoc khi nhan job (atomic) -> ko du thi tu choi 402. Worker hoan lai neu loi."""
    if not req.script:
        raise HTTPException(status_code=400, detail="Script trống")
    _cleanup_jobs()
    if _job_q.qsize() >= MAX_QUEUE:
        raise HTTPException(
            status_code=503,
            detail="Hệ thống đang quá tải, nhiều video đang chờ. Thử lại sau ít phút nhé."
        )
    # Tru credit truoc — neu het, KHONG nhan job (tranh tao video roi moi phat hien het tien).
    if not reserve_credit(uid, VIDEO_CREDIT_COST):
        raise HTTPException(
            status_code=402,
            detail="Bạn đã hết credit. Nâng cấp gói để tạo thêm video nhé."
        )
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending", "error": None, "path": None,
            "ts": time.time(), "uid": uid, "cost": VIDEO_CREDIT_COST,
        }
        _job_reqs[job_id] = req
    _job_q.put(job_id)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/video-status/{job_id}")
def video_status(job_id: str):
    """Hoi trang thai job: pending | processing | done | error.
    Khi pending, tra them `position` = vi tri trong hang doi (1 = ke tiep)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job không tồn tại hoặc đã hết hạn")
        status, error = job["status"], job["error"]
    position = None
    if status == "pending":
        try:
            position = list(_job_q.queue).index(job_id) + 1
        except ValueError:
            position = None
    return {"status": status, "error": error, "position": position}


@app.get("/api/video-result/{job_id}")
def video_result(job_id: str):
    """Tai MP4 khi job xong (doc tu disk). File tu het han theo JOB_TTL."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job không tồn tại hoặc đã hết hạn")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job["error"] or "Lỗi tạo video")
    if job["status"] != "done" or not job.get("path") or not os.path.exists(job["path"]):
        raise HTTPException(status_code=409, detail="Video chưa sẵn sàng")
    return FileResponse(
        job["path"], media_type="video/mp4", filename="tiktok_video.mp4"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
