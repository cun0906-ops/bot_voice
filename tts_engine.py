import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".model_vieneu_cache"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))

from vieneu import Vieneu
import vieneu_utils.core_utils as core_utils

# Các cấu hình ngắt nghỉ và ranh giới câu (tương thích trực tiếp với silence_p và max_chars của VieNeu v3 Turbo)
PAUSE_STYLE_CONFIG = {
    "default": {"silence_p": 0.15, "max_chars": 256},      # Mặc định: Giữ nguyên vẹn 100% nhịp điệu gốc của model
    "natural": {"silence_p": 0.15, "max_chars": 256},      # Tự nhiên, cân bằng
    "relaxed": {"silence_p": 0.40, "max_chars": 130},      # Thư thả, sâu lắng (sách nói, podcast đêm)
    "expressive": {"silence_p": 0.30, "max_chars": 120},   # Diễn cảm, rõ vế (kể chuyện)
    "presentation": {"silence_p": 0.25, "max_chars": 140}, # Dõng dạc, rành mạch (thuyết trình, bài giảng)
    "dramatic": {"silence_p": 0.55, "max_chars": 110},     # Lắng đọng, kịch tính (phim ảnh, tâm trạng)
    "compact": {"silence_p": 0.08, "max_chars": 256},      # Nhanh, liền mạch (tin tức, video ngắn)
    "rapid": {"silence_p": 0.02, "max_chars": 300},        # Siêu tốc, nhịp dồn (quảng cáo xả hàng, tin nóng)
}

PAUSE_REGEX = re.compile(r"\[(?:ngh[iỉ]|pause):([0-9.]+)s?\]", re.IGNORECASE)


def normalize_vietnamese_text(text: str, custom_dict: Optional[Dict[str, str]] = None) -> str:
    """Tự động chuẩn hóa từ viết tắt, số tiền, ngày tháng, đơn vị đo lường, thương hiệu và từ điển tùy chỉnh."""
    if not text:
        return ""

    # 0. Áp dụng từ điển tùy chỉnh nếu người dùng cung cấp
    if custom_dict and isinstance(custom_dict, dict):
        for k, v in custom_dict.items():
            if k and v:
                pattern = r"\b" + re.escape(str(k).strip()) + r"\b"
                text = re.sub(pattern, str(v).strip(), text, flags=re.IGNORECASE)

    # 1. Số điện thoại & Hotline Việt Nam (đọc từng chữ số rõ ràng)
    def _digits_to_vi(match):
        digits = re.sub(r"\D", "", match.group(0))
        d_map = {
            "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
            "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín"
        }
        return " ".join(d_map.get(d, d) for d in digits)

    text = re.sub(r"\b(1900|1800)\s*(\d{4})\b", r"tổng đài \1 \2", text)
    text = re.sub(r"\b(0[35789]\d{8})\b", _digits_to_vi, text)
    text = re.sub(r"\b(0[35789]\d{1,2}[-.\s]\d{3}[-.\s]\d{3,4})\b", _digits_to_vi, text)

    # 2. Ngày tháng dạng DD/MM/YYYY hoặc DD/MM
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", r"ngày \1 tháng \2 năm \3", text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", r"ngày \1 tháng \2", text)

    # 3. Giờ giấc
    text = re.sub(r"\b(\d{1,2})h(\d{1,2})\b", r"\1 giờ \2 phút", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d{1,2})h\b", r"\1 giờ", text, flags=re.IGNORECASE)

    # 4. Đơn vị tiền tệ & số viết tắt
    text = re.sub(r"(\d+)\s*(?:k|K)\b", r"\1 nghìn", text)
    text = re.sub(r"(\d+)\s*(?:tr|TR|trđ|Trđ)\b", r"\1 triệu", text)
    text = re.sub(r"(\d+)[,\.](\d+)\s*(?:tr|TR|trieu|triệu)\b", r"\1 triệu \2", text)
    text = re.sub(r"(\d+)\s*tỷ\b", r"\1 tỷ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:đ|vnđ|vnd|đồng)\b", r"\1 đồng", text, flags=re.IGNORECASE)
    text = re.sub(r"\$(\d+)", r"\1 đô la", text)
    text = re.sub(r"(\d+)\s*\$", r"\1 đô la", text)
    text = re.sub(r"(\d+)\s*%", r"\1 phần trăm", text)

    # 5. Thế kỷ số La Mã
    text = re.sub(r"\bthế kỷ XXI\b", "thế kỷ hai mươi mốt", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthế kỷ XX\b", "thế kỷ hai mươi", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthế kỷ XIX\b", "thế kỷ mười chín", text, flags=re.IGNORECASE)

    # 6. Đơn vị đo lường
    text = re.sub(r"(\d+)\s*km/h\b", r"\1 ki lô mét trên giờ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*km\b", r"\1 ki lô mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:m2|m²)\b", r"\1 mét vuông", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:m3|m³)\b", r"\1 mét khối", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*kg\b", r"\1 ki lô gam", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*cm\b", r"\1 xen ti mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*mm\b", r"\1 mi li mét", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)\s*(?:gb|GB)\b", r"\1 ghi ga bai", text)
    text = re.sub(r"(\d+)\s*(?:mb|MB)\b", r"\1 mê ga bai", text)

    # 7. Từ viết tắt tiếng Anh đặc thù & đọc từng chữ cái tiếng Anh cho từ IN HOA (KOC, KOL, KPI, ROI...)
    EN_LETTER_MAP = {
        'A': 'ây', 'B': 'bi', 'C': 'xi', 'D': 'đi', 'E': 'i',
        'F': 'ép', 'G': 'gi', 'H': 'hét', 'I': 'ai', 'J': 'dây',
        'K': 'cây', 'L': 'eo', 'M': 'em', 'N': 'en', 'O': 'âu',
        'P': 'pi', 'Q': 'kiu', 'R': 'a', 'S': 'ét', 'T': 'ti',
        'U': 'du', 'V': 'vi', 'W': 'đáp-liu', 'X': 'ích', 'Y': 'oai', 'Z': 'dét'
    }

    # Danh mục viết tắt Digital Marketing / KOC / KOL / Tech phổ biến
    curated_acronyms = [
        (r"\bKOCs\b", "cây-âu-xi"),
        (r"\bKOC\b", "cây-âu-xi"),
        (r"\bKOLs\b", "cây-âu-eo"),
        (r"\bKOL\b", "cây-âu-eo"),
        (r"\bKPIs\b", "cây-pi-ai"),
        (r"\bKPI\b", "cây-pi-ai"),
        (r"\bROI\b", "a-âu-ai"),
        (r"\bROAS\b", "a-âu-ây-ét"),
        (r"\bUGC\b", "du-gi-xi"),
        (r"\bMCN\b", "em-xi-en"),
        (r"\bCPM\b", "xi-pi-em"),
        (r"\bCPC\b", "xi-pi-xi"),
        (r"\bCTR\b", "xi-ti-a"),
        (r"\bCTA\b", "xi-ti-ây"),
        (r"\bGMV\b", "gi-em-vi"),
        (r"\bSEO\b", "ét-i-âu"),
        (r"\bB2B\b", "bi-tu-bi"),
        (r"\bB2C\b", "bi-tu-xi"),
        (r"\bD2C\b", "đi-tu-xi"),
        (r"\bPR\b", "pi-a"),
        (r"\bHR\b", "hét-a"),
        (r"\bQA\b", "kiu-ây"),
        (r"\bQC\b", "kiu-xi"),
        (r"\bCEO\b", "xi-i-âu"),
        (r"\bCFO\b", "xi-ép-âu"),
        (r"\bCMO\b", "xi-em-âu"),
        (r"\bCTO\b", "xi-ti-âu"),
        (r"\bCOO\b", "xi-âu-âu"),
        (r"\bVIP\b", "vi-ai-pi"),
        (r"\bIT\b", "ai-ti"),
        (r"\bUI\b", "du-ai"),
        (r"\bUX\b", "du-ích"),
        (r"\bAI\b", "ây-ai"),
        (r"\bMC\b", "em-xi"),
        (r"\bDJ\b", "đi-dây"),
        (r"\bSKU\b", "ét-cây-du"),
        (r"\bAPI\b", "ây-pi-ai"),
        (r"\bSDK\b", "ét-đi-cây"),
        (r"\bSMS\b", "ét-em-ét"),
        (r"\bOTP\b", "âu-ti-pi"),
        (r"\bVAT\b", "vi-ây-ti"),
        (r"\bFAQ\b", "ép-ây-kiu"),
        (r"\bDIY\b", "đi-ai-oai"),
        (r"\bDNA\b", "đi-en-ây"),
        (r"\bGPS\b", "gi-pi-ét"),
        (r"\bATM\b", "ây-ti-em"),
        (r"\bPOS\b", "pi-âu-ét"),
        (r"\bID\b", "ai-đi"),
        (r"\bCV\b", "xi-vi"),
        (r"\bPC\b", "pi-xi"),
        (r"\bUSB\b", "du-ét-bi"),
        (r"\bOOH\b", "âu-âu-hét"),
        (r"\bOTT\b", "âu-ti-ti"),
        (r"\bPBN\b", "pi-bi-en"),
    ]
    for pattern, rep in curated_acronyms:
        text = re.sub(pattern, rep, text)

    # Tự động đọc từng chữ cái tiếng Anh cho CÁC TỪ VIẾT HOA CÒN LẠI (2-5 ký tự IN HOA)
    # Trừ các từ viết tắt tiếng Việt cơ quan / nhà nước
    VIETNAMESE_UPPERCASE_EXCLUDE = {
        "STK", "SĐT", "TP", "UBND", "HĐND", "BHYT", "BHXH", "CCCD", "CMND", "CSKH",
        "KTS", "BS", "THS", "TS", "PGS", "GS", "PCCC", "TNMT", "GTVT", "THPT", "THCS",
        "ĐH", "CĐ", "VND", "VNĐ", "CLB", "TNHH", "CP", "DN", "BGD", "BYT", "BCA", "BQP",
        "BCT", "TCVN", "BTTTT", "VTV", "HTV", "VOV", "TTXVN", "NXB", "HĐQT"
    }

    def _spell_english_acronym(m):
        word = m.group(1)
        if word in VIETNAMESE_UPPERCASE_EXCLUDE:
            return word
        letters = [EN_LETTER_MAP.get(c, c) for c in word if c in EN_LETTER_MAP]
        if len(letters) == len(word):
            return "-".join(letters)
        return word

    text = re.sub(r"\b([A-Z]{2,5})\b", _spell_english_acronym, text)

    # 8. Từ viết tắt xưng danh, nghiệp vụ tiếng Việt
    abbreviations = [
        (r"\bSTK\b", "số tài khoản"),
        (r"\bSĐT\b", "số điện thoại"),
        (r"\bđ/c\b", "địa chỉ"),
        (r"\bKTS\b", "kiến trúc sư"),
        (r"\bBS\b", "bác sĩ"),
        (r"\bThS\b", "thạc sĩ"),
        (r"\bTS\b", "tiến sĩ"),
        (r"\bPGS\b", "phó giáo sư"),
        (r"\bGS\b", "giáo sư"),
        (r"\bsp\b", "sản phẩm"),
        (r"\bko\b", "không"),
        (r"\bk\b", "không"),
        (r"\bđc\b", "được"),
        (r"\bvs\b", "với"),
        (r"\bib\b", "in-bóc"),
    ]
    for pattern, rep in abbreviations:
        text = re.sub(pattern, rep, text, flags=re.IGNORECASE)

    # 9. Từ mượn công nghệ / thương hiệu thông dụng
    brand_map = [
        (r"\b(?:ChatGPT|chatgpt)\b", "chát giê-pi-ti"),
        (r"\b(?:GPT|gpt)\b", "giê-pi-ti"),
        (r"\b(?:FB|fb|Facebook|facebook)\b", "phây-búc"),
        (r"\b(?:Zalo|zalo)\b", "da-lô"),
        (r"\b(?:TikTok|tiktok)\b", "tíc-tốc"),
        (r"\b(?:YouTube|youtube|yt|YT)\b", "diu-túp"),
        (r"\b(?:Website|website|web)\b", "oép-sai"),
        (r"\b(?:Hotline|hotline)\b", "hót-lai"),
        (r"\b(?:Email|email|mail)\b", "ê-meo"),
        (r"\b(?:Wifi|wifi)\b", "oai-phai"),
        (r"\b(?:Livestream|livestream)\b", "lai-chim"),
        (r"\b(?:Admin|admin)\b", "át-min"),
        (r"\b(?:Freeship|freeship)\b", "phi-síp"),
        (r"\b(?:Ship|ship)\b", "síp"),
        (r"\b(?:Sale|sale)\b", "seo"),
        (r"\b(?:Review|review)\b", "ri-viu"),
        (r"\b(?:Video|video)\b", "vi-de-ô"),
        (r"\b(?:Link|link)\b", "linh"),
        (r"\b(?:App|app)\b", "áp"),
    ]
    for pattern, replacement in brand_map:
        text = re.sub(pattern, replacement, text)

    return text


def preprocess_natural_text(
    text: str,
    auto_normalize: bool = True,
    custom_dict: Optional[Dict[str, str]] = None,
) -> str:
    """Chuẩn hóa văn bản, hỗ trợ thẻ lấy hơi, nhịp thở, thẻ nhấn mạnh và khoảng ngắt nghỉ tự nhiên."""
    if not text:
        return ""

    if auto_normalize:
        text = normalize_vietnamese_text(text, custom_dict=custom_dict)

    # 1. Chuyển đổi các thẻ nhấn nhá:
    # [nhấn]từ ngữ[/nhấn] hoặc [nhấn: từ ngữ] hoặc **từ ngữ**
    # Kỹ thuật tạo trọng âm: Thêm khoảng ngắt nhẹ " ... " trước từ và " — " sau từ để AI nhấn mạnh trọng âm
    text = re.sub(r"\[(?:nhấn|stress|em(?:phasize)?):?\s*([^\]]+)\]", r" ... \1 — ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(?:nhấn|stress|em(?:phasize)?)\](.*?)\[/(?:nhấn|stress|em(?:phasize)?)\]", r" ... \1 — ", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*([^*]+)\*\*", r" ... \1 — ", text)

    # 2. Chuyển đổi các thẻ lấy hơi / thở tự nhiên sang [thở dài] (emotion token 2 của VieNeu sinh âm thanh hít thở thật)
    text = re.sub(
        r"\[(thở|lay hoi|lấy hơi|hít thở|hit tho|tho|breathe|hơi|thở phào)\]",
        "[thở dài]",
        text,
        flags=re.IGNORECASE,
    )

    # 3. Chuyển đổi thẻ [cười] (emotion token 1 của VieNeu)
    text = re.sub(
        r"\[(cười mỉm|cười vui|cuoi|chuckle|laugh)\]",
        "[cười]",
        text,
        flags=re.IGNORECASE,
    )

    # 4. Chuyển đổi thẻ [hắng giọng] (emotion token 3 của VieNeu)
    text = re.sub(
        r"\[(hắng|hang giong|clear throat|cough)\]",
        "[hắng giọng]",
        text,
        flags=re.IGNORECASE,
    )

    # 5. Chuyển đổi các thẻ nghỉ không kèm số giây sang dấu ngắt nhịp tự nhiên
    text = re.sub(r"\[(nghỉ ngắn|nghi ngan|short pause)\]", " ... ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[(nghỉ dài|nghi dai|long pause)\]", " .\n\n", text, flags=re.IGNORECASE)

    # 6. Chuẩn hóa gạch ngang và dấu ba chấm để AI tạo khoảng ngập ngừng tự nhiên
    text = re.sub(r"\s*--\s*", " — ", text)
    text = re.sub(r"\.{4,}", "...", text)

    return text.strip()


def apply_auto_phrasing(text: str, style: str = "natural") -> str:
    """Tự động phân tích và thêm dấu câu, điểm lấy hơi, khoảng ngắt nghỉ luyến láy và từ nhấn nhá
    để giọng AI đọc có ngữ điệu truyền cảm, tự nhiên như người thật."""
    if not text or not text.strip():
        return ""

    t = text.strip()

    # 1. Thêm [lấy hơi] ở đầu nếu là câu chào hoặc mở đầu bài
    if not re.match(r"^\[(lấy hơi|thở|thở dài|cười|hắng giọng)\]", t, flags=re.IGNORECASE):
        greeting_pattern = r"^(dạ\s+)?(xin\s+chào|chào\s+quý\s+vị|chào\s+các\s+bạn|kính\s+chào|chào\s+mừng|chào\s+mọi\s+người|hello|hi)\b"
        if re.match(greeting_pattern, t, flags=re.IGNORECASE):
            t = "[lấy hơi] " + t
        elif style in ["story", "podcast", "expressive"]:
            t = "[lấy hơi] " + t

    # 2. Xử lý các liên từ dẫn dắt & trạng từ tạo điểm nhấn (Discourse Markers & Connectors)
    # Thêm dấu luyến láy (...) hoặc ngắt vế ( — ) để AI hạ nhịp rồi nhấn vế sau
    connectors_pause = [
        (r"\b(tuy nhiên|nhưng mà|thế nhưng|đáng chú ý là|sự thật là|trên thực tế)\b(?!\s*[\.,!?:…—])", r"\1..."),
        (r"\b(đặc biệt là|chính vì thế|hơn thế nữa|không chỉ vậy|không những thế|quan trọng nhất là|điều tuyệt vời là)\b(?!\s*[\.,!?:…—])", r"\1 —"),
        (r"\b(chắc chắn rằng|hãy nhớ rằng|bạn có biết rằng|hãy tưởng tượng)\b(?!\s*[\.,!?:…—])", r"\1..."),
        (r"\b(tóm lại|nói cách khác|nhìn chung|kết quả là)\b(?!\s*[\.,!?:…—])", r"\1,"),
    ]
    for pattern, rep in connectors_pause:
        t = re.sub(pattern, rep, t, flags=re.IGNORECASE)

    # 3. Nhấn nhá theo phong cách cụ thể
    if style in ["sales", "rapid"]:
        # Phong cách bán hàng / TikTok / Review: Hào hứng, dồn dập, nhấn mạnh ưu đãi & hành động
        cta_words = [
            (r"\b(KOCs?|KOLs?)\b", r" **\1**"),
            (r"\b(ngay bây giờ|ngay hôm nay|chỉ hôm nay|đừng bỏ lỡ|nhanh tay|duy nhất hôm nay)\b(?!\s*[\.,!?:…—])", r" — **\1**!"),
            (r"\b(hoàn toàn miễn phí|freeship toàn quốc|bảo hành trọn đời|giảm ngay|ưu đãi cực khủng)\b(?!\s*[\.,!?:…—])", r" **\1**,"),
            (r"\b(gấp\s*\d+\s*lần|\d+%\s*giảm|giảm\s*\d+%)\b(?!\s*[\.,!?:…—])", r" **\1**!"),
        ]
        for pattern, rep in cta_words:
            t = re.sub(pattern, rep, t, flags=re.IGNORECASE)
        # Câu kết nếu chưa có dấu chấm than thì thêm ! để kết dứt khoát
        if not re.search(r"[.!?…]$", t):
            t += "!"
    elif style in ["story", "expressive"]:
        # Phong cách kể chuyện / diễn cảm: Sâu lắng, ngập ngừng, luyến láy
        story_words = [
            (r"\b(ngày xửa ngày xưa|một ngày nọ|bỗng nhiên|thế rồi|sau tất cả|từ sâu thẳm)\b(?!\s*[\.,!?:…—])", r"\1..."),
            (r"\b(lạ lùng thay|bất ngờ thay|thật không ngờ|không ai ngờ rằng)\b(?!\s*[\.,!?:…—])", r"\1 —"),
        ]
        for pattern, rep in story_words:
            t = re.sub(pattern, rep, t, flags=re.IGNORECASE)
    elif style == "podcast":
        # Phong cách podcast đêm / tâm sự: Thư thả, lắng đọng, thêm khoảng thở
        podcast_words = [
            (r"\b(đôi khi|có những ngày|bạn biết không|thực ra thì|trong cuộc sống)\b(?!\s*[\.,!?:…—])", r"\1..."),
            (r"\b(hãy lắng nghe|dành một phút|nghĩ lại xem)\b(?!\s*[\.,!?:…—])", r"\1 [nghỉ:0.5s]"),
        ]
        for pattern, rep in podcast_words:
            t = re.sub(pattern, rep, t, flags=re.IGNORECASE)

    # 4. Tách nhịp tự nhiên cho các câu quá dài (> 16 từ mà không có dấu ngắt nghỉ)
    def split_long_clauses(match):
        sent = match.group(0)
        words = sent.split()
        if len(words) > 16 and not re.search(r"[,;—…]", sent):
            sub_clause_pattern = r"\s+(mà|để|nhằm|khiến cho|giúp cho|đồng thời|với mong muốn|trong khi|bởi vì)\s+"
            sent = re.sub(sub_clause_pattern, r", \1 ", sent, count=1)
        return sent

    t = re.sub(r"[^.!?…\n]+[.!?…\n]?", split_long_clauses, t)

    # 5. Làm sạch khoảng trắng thừa và dấu câu trùng lặp
    t = re.sub(r"\s+([.,!?:;])", r"\1", t)
    t = re.sub(r",\s*,+", ",", t)
    t = re.sub(r"\.{4,}", "...", t)
    t = re.sub(r"[ \t]+", " ", t)

    return t.strip()


def split_text_into_chunks(text: str, max_words: int = 35) -> List[str]:
    """Chia văn bản thành các câu hoặc nhóm câu nhỏ gọn (~20-35 từ) để mô hình không bị quá tải trên bài dài."""
    clean = text.strip()
    if not clean:
        return []

    # Chia theo dấu kết thúc câu hoặc xuống dòng
    raw_sentences = re.split(r"(?<=[.!?…\n])\s+", clean)
    chunks = []
    current_chunk = []
    current_words = 0

    for sent in raw_sentences:
        s = sent.strip()
        if not s:
            continue
        words_in_s = len(s.split())
        if current_words > 0 and (current_words + words_in_s > max_words):
            chunks.append(" ".join(current_chunk))
            current_chunk = [s]
            current_words = words_in_s
        else:
            current_chunk.append(s)
            current_words += words_in_s

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [clean]


def wsola_time_stretch(audio: np.ndarray, rate: float, sr: int = 48000) -> np.ndarray:
    """Co giãn thời gian bằng WSOLA trong miền thời gian (Waveform Similarity Overlap-Add).
    Khắc phục triệt để hiện tượng ồm ồm, méo tiếng hoặc vang hẫng của Phase Vocoder.
    Giữ nguyên 100% âm sắc, formants và độ tròn vành rõ chữ của giọng AI.
    """
    if abs(rate - 1.0) < 0.02 or len(audio) == 0:
        return audio

    rate = float(max(0.65, min(1.75, rate)))
    win_size = int(0.025 * sr)
    if win_size % 2 != 0:
        win_size += 1
    syn_hop = win_size // 2
    ana_hop = int(round(syn_hop * rate))
    delta = int(0.007 * sr)

    window = np.hanning(win_size).astype(np.float32)
    total_samples = len(audio)

    if total_samples < win_size + delta * 2:
        return audio

    est_out_len = int(total_samples / rate) + win_size * 2
    out_audio = np.zeros(est_out_len, dtype=np.float32)
    norm_window = np.zeros(est_out_len, dtype=np.float32)

    # Khởi tạo frame đầu
    out_audio[0:win_size] += audio[0:win_size] * window
    norm_window[0:win_size] += window

    out_pos = syn_hop
    in_pos = ana_hop
    prev_target_pos = syn_hop

    from scipy.signal import correlate

    while (in_pos + win_size + delta < total_samples) and (out_pos + win_size < est_out_len):
        search_start = max(0, in_pos - delta)
        search_end = min(total_samples, in_pos + delta + win_size)

        if search_end - search_start < win_size:
            chosen_seg = audio[in_pos : in_pos + win_size]
            prev_target_pos = in_pos + syn_hop
        else:
            search_region = audio[search_start:search_end]
            ref_seg = audio[prev_target_pos : prev_target_pos + win_size]

            if len(ref_seg) == win_size and len(search_region) >= win_size:
                corr = correlate(search_region, ref_seg, mode="valid")
                best_idx = int(np.argmax(corr))
                chosen_start = search_start + best_idx
                chosen_seg = audio[chosen_start : chosen_start + win_size]
                prev_target_pos = chosen_start + syn_hop
            else:
                chosen_seg = audio[in_pos : in_pos + win_size]
                prev_target_pos = in_pos + syn_hop

        out_audio[out_pos : out_pos + win_size] += chosen_seg * window
        norm_window[out_pos : out_pos + win_size] += window

        out_pos += syn_hop
        in_pos += ana_hop

    valid_len = out_pos + win_size
    mask = norm_window[:valid_len] > 1e-4
    out_audio[:valid_len][mask] /= norm_window[:valid_len][mask]

    return out_audio[:valid_len]


def clean_pitch_shift(audio: np.ndarray, n_steps: float, sr: int = 48000) -> np.ndarray:
    """Dịch cao độ giọng nói mà KHÔNG gây méo tiếng hay ồm ồm như phase vocoder."""
    if abs(n_steps) < 0.25 or len(audio) == 0:
        return audio

    n_steps = float(max(-3.5, min(3.5, n_steps)))
    factor = 2.0 ** (n_steps / 12.0)

    try:
        target_sr = int(round(sr / factor))
        try:
            import soxr
            resampled = soxr.resample(audio, in_rate=sr, out_rate=target_sr, quality="HQ")
        except Exception:
            try:
                import librosa
                resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            except Exception:
                from scipy.signal import resample_poly
                from fractions import Fraction
                frac = Fraction(target_sr, sr).limit_denominator(100)
                resampled = resample_poly(audio, frac.numerator, frac.denominator).astype(np.float32)

        shifted = wsola_time_stretch(resampled, rate=factor, sr=sr)
    except Exception as e:
        print(f"⚠️ Không thể dịch cao độ giọng: {e}")
        shifted = audio

    # Khử dải ù trầm nếu hạ tông (loại bỏ tiếng ồm ở 80Hz - 250Hz)
    if n_steps < 0 and len(shifted) > 0:
        try:
            from scipy.signal import butter, sosfilt
            sos = butter(2, 80, btype="highpass", fs=sr, output="sos")
            shifted = sosfilt(sos, shifted).astype(np.float32)
        except Exception:
            pass

    return shifted


def clean_special_characters(text: str) -> str:
    """Tự động loại bỏ icon, emoji, ký tự lạ để AI không đọc thành tiếng rác."""
    if not text:
        return ""
    # Bảo vệ các thẻ như [lấy hơi], [nghỉ:0.5s], [thở dài], [nhấn], **...**
    tags = []
    def save_tag(m):
        tags.append(m.group(0))
        return f"___TAG_{len(tags)-1}___"

    text = re.sub(r"\[(?:ngh[iỉ]|pause|thở|thở dài|lấy hơi|cười|hắng giọng|thì thầm|nghỉ ngắn|nghỉ dài|nhấn|stress|em)[^\]]*\]", save_tag, text, flags=re.IGNORECASE)
    text = re.sub(r"\[/(?:nhấn|stress|em)\]", save_tag, text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*[^*]+\*\*", save_tag, text)

    # Loại bỏ emoji unicode
    emoji_pattern = re.compile(
        "[\U00010000-\U0010ffff]|"
        "[\u2600-\u26ff]|[\u2700-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u203c-\u2049]|[\u25aa-\u25fe]",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub(" ", text)

    # Loại bỏ các ký tự rác lạ (@, #, ^, &, ~, `, {, }, |, <, >, =) - giữ lại *
    text = re.sub(r"[@#^&~`|{}<>=_\\]", " ", text)
    # Rút gọn khoảng trắng
    text = re.sub(r"[ \t]+", " ", text).strip()

    # Khôi phục các thẻ
    for i, tag in enumerate(tags):
        text = text.replace(f"___TAG_{i}___", tag)

    return text


def format_srt_time(seconds: float) -> str:
    """Đổi số giây thành định dạng timecode chuẩn SubRip SRT: HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt_content(cues: List[Dict[str, Any]]) -> str:
    """Tạo nội dung file .SRT chuẩn từ danh sách các câu kèm mốc thời gian."""
    lines = []
    for i, cue in enumerate(cues, 1):
        start = format_srt_time(cue["start"])
        end = format_srt_time(cue["end"])
        text = cue["text"].strip()
        # Loại bỏ các thẻ điều khiển nội bộ và thẻ nhấn khỏi phụ đề hiển thị
        clean_sub_text = re.sub(r"\[(?:nhấn|stress|em(?:phasize)?)\](.*?)\[/(?:nhấn|stress|em(?:phasize)?)\]", r"\1", text, flags=re.IGNORECASE)
        clean_sub_text = re.sub(r"\[(?:nhấn|stress|em(?:phasize)?):?\s*([^\]]+)\]", r"\1", clean_sub_text, flags=re.IGNORECASE)
        clean_sub_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean_sub_text)
        clean_sub_text = re.sub(r"\[[^\]]+\]", "", clean_sub_text).strip()
        if not clean_sub_text:
            clean_sub_text = text
        lines.append(f"{i}\n{start} --> {end}\n{clean_sub_text}\n")
    return "\n".join(lines)


def split_text_to_subtitles(text: str, max_chars: int = 200) -> List[Dict[str, Any]]:
    """Chia văn bản thành các câu phụ đề ngắn gọn, nhận diện mốc nghỉ tùy biến [nghỉ:Xs]."""
    clean = text.strip()
    if not clean:
        return []

    # Tách văn bản theo các thẻ ngắt nghỉ dạng [nghỉ:0.5s] hoặc [pause:1s]
    pause_tag_pattern = re.compile(r"(\[(?:ngh[iỉ]|pause):(?:[0-9.]+)s?\])", re.IGNORECASE)
    parts = pause_tag_pattern.split(clean)

    result: List[Dict[str, Any]] = []

    for part in parts:
        part_str = part.strip()
        if not part_str:
            continue

        pause_match = re.match(r"^\[(?:ngh[iỉ]|pause):([0-9.]+)s?\]$", part_str, flags=re.IGNORECASE)
        if pause_match:
            try:
                pause_sec = float(pause_match.group(1))
                pause_sec = max(0.05, min(10.0, pause_sec))
            except (ValueError, TypeError):
                pause_sec = 0.5

            if result:
                result[-1]["custom_pause"] = pause_sec
            else:
                result.append({
                    "text": "",
                    "pause_type": "custom",
                    "custom_pause": pause_sec,
                })
            continue

        # Phần văn bản thông thường
        raw_sentences = re.split(r"(?<=[.!?…\n])\s+", part_str)
        cues_text = []

        for s in raw_sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            if len(s_clean) > max_chars:
                sub_parts = re.split(r"(?<=[,;:])\s+", s_clean)
                cur = []
                cur_len = 0
                for p in sub_parts:
                    p_len = len(p)
                    if cur_len > 0 and (cur_len + p_len > max_chars):
                        cues_text.append(" ".join(cur))
                        cur = [p]
                        cur_len = p_len
                    else:
                        cur.append(p)
                        cur_len += p_len
                if cur:
                    cues_text.append(" ".join(cur))
            else:
                cues_text.append(s_clean)

        for item in cues_text:
            trimmed = item.strip()
            if not trimmed:
                continue
            pause_type = "sentence"
            if trimmed.endswith((",", ";", ":")):
                pause_type = "comma"
            elif trimmed.endswith((".", "!", "?", "…")):
                pause_type = "sentence"
            else:
                pause_type = "segment"

            result.append({
                "text": trimmed,
                "pause_type": pause_type,
                "custom_pause": None,
            })

    return result


class VoiceCloner:
    def __init__(self, threads: int = 4):
        self.device = "cpu (ONNX Runtime)"
        self._tts = Vieneu(mode="v3turbo", backend="onnx", threads=threads)
        self.sample_rate = getattr(self._tts, "sample_rate", 48000)
        self._registered_references = {}
        self._generation_lock = threading.Lock()

    def get_preset_voices(self) -> Dict[str, Dict[str, Any]]:
        """Lấy danh mục các giọng mẫu có sẵn của VieNeu v3 Turbo kèm metadata."""
        voices = {}
        for name, data in self._tts._preset_voices.items():
            voices[name] = {
                "name": name,
                "description": data.get("description", ""),
                "gender": data.get("gender", ""),
                "region": data.get("region", ""),
                "style": data.get("style", "tu_nhien"),
                "is_clone": False,
            }
        return voices

    def register_custom_voice(
        self,
        voice_id: str,
        audio_path: str,
        description: str = "",
        gender: str = "",
    ) -> str:
        """Đăng ký giọng clone mới từ file âm thanh người dùng tải lên."""
        audio_file = Path(audio_path).resolve()
        if not audio_file.is_file():
            raise FileNotFoundError(f"Không tìm thấy file audio: {audio_file}")

        voice_name = f"clone-{voice_id}"
        with self._generation_lock:
            self._tts.add_voice(
                name=voice_name,
                ref_audio=audio_file,
                denoise=True,
                use_ref_codes=True,
                description=description,
                gender=gender,
                save=False,
            )
            self._registered_references[voice_name] = audio_file
        return voice_name

    def remove_custom_voice(self, voice_id: str) -> None:
        """Hủy đăng ký giọng clone."""
        voice_name = f"clone-{voice_id}"
        with self._generation_lock:
            if hasattr(self._tts, "remove_voice"):
                self._tts.remove_voice(voice_name)
            self._registered_references.pop(voice_name, None)

    def _infer_segment(
        self,
        segment_text: str,
        voice_name: str,
        temperature: float,
        dynamic_max_frames: int,
        silence_p: float = 0.15,
        max_chars: int = 256,
        auto_normalize: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Sinh audio cho một phân đoạn văn bản."""
        clean_text = preprocess_natural_text(
            segment_text, auto_normalize=auto_normalize, custom_dict=custom_dict
        )
        if not clean_text:
            return np.array([], dtype=np.float32)

        audio = self._tts.infer(
            clean_text,
            voice=voice_name,
            temperature=temperature,
            top_k=25,
            top_p=0.95,
            repetition_penalty=1.2,
            max_new_frames=dynamic_max_frames,
            max_chars=max_chars,
            silence_p=silence_p,
        )
        return audio if audio is not None else np.array([], dtype=np.float32)

    def _infer_block_smart(
        self,
        block_text: str,
        voice_name: str,
        temperature: float,
        silence_p: float = 0.15,
        max_chars: int = 256,
        auto_normalize: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> np.ndarray:
        """Xử lý khối chữ: nếu ngắn thì infer trực tiếp, nếu dài thì chia câu thông minh để audio không bao giờ bị cắt ngắn."""
        words = block_text.strip().split()
        if len(words) <= 40:
            frame_cap = max(350, min(2200, len(words) * 50))
            return self._infer_segment(
                block_text, voice_name, temperature, frame_cap,
                silence_p=silence_p, max_chars=max_chars,
                auto_normalize=auto_normalize, custom_dict=custom_dict,
            )

        chunks = split_text_into_chunks(block_text, max_words=32)
        sub_audios: List[np.ndarray] = []
        silence_samples = int(0.18 * self.sample_rate)
        gap_zeros = np.zeros(silence_samples, dtype=np.float32)

        for chunk in chunks:
            c_words = len(chunk.split())
            frame_cap = max(350, min(1800, c_words * 50))
            audio_part = self._infer_segment(
                chunk, voice_name, temperature, frame_cap,
                silence_p=silence_p, max_chars=max_chars,
                auto_normalize=auto_normalize, custom_dict=custom_dict,
            )
            if len(audio_part) > 0:
                if sub_audios:
                    sub_audios.append(gap_zeros)
                sub_audios.append(audio_part)

        if not sub_audios:
            return np.array([], dtype=np.float32)
        return np.concatenate(sub_audios)

    def generate(
        self,
        gen_text: str,
        voice_id: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: str = "",
        output_path: str = "output.wav",
        output_srt_path: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        stability: float = 50.0,
        similarity: float = 75.0,
        speaker_boost: bool = True,
        temperature: Any = "default",
        pause_style: str = "default",
        comma_pause: float = 0.3,
        sentence_pause: float = 0.3,
        segment_pause: float = 0.3,
        strip_special_chars: bool = False,
        max_chars_per_line: int = 200,
        normalize_text: bool = True,
        generate_srt: bool = True,
        custom_dict: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not gen_text or not gen_text.strip():
            raise ValueError("Nội dung văn bản không được để trống.")

        if strip_special_chars:
            gen_text = clean_special_characters(gen_text)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_srt_path is None:
            output_srt_path = str(output_file.with_suffix(".srt"))

        pause_config = PAUSE_STYLE_CONFIG.get(pause_style, PAUSE_STYLE_CONFIG["default"])
        silence_p = pause_config["silence_p"]
        max_chars = pause_config["max_chars"]

        with self._generation_lock:
            # 1. Trường hợp sử dụng file âm thanh tham chiếu (Custom Voice Clone)
            if ref_audio_path:
                reference_path = Path(ref_audio_path).resolve()
                if not reference_path.is_file():
                    raise FileNotFoundError(f"Không tìm thấy audio tham chiếu: {reference_path}")

                voice_name = f"clone-{reference_path.stem}"
                if self._registered_references.get(voice_name) != reference_path:
                    self._tts.add_voice(
                        voice_name,
                        reference_path,
                        denoise=True,
                        use_ref_codes=True,
                        save=False,
                    )
                    self._registered_references[voice_name] = reference_path
            # 2. Trường hợp sử dụng giọng mẫu có sẵn (Preset Voice hoặc đã đăng ký)
            elif voice_id:
                resolved = self._tts.resolve_voice_name(voice_id) or voice_id
                if resolved not in self._tts._preset_voices:
                    clone_candidate = f"clone-{voice_id}"
                    if clone_candidate in self._tts._preset_voices:
                        resolved = clone_candidate
                    else:
                        raise ValueError(f"Không tìm thấy giọng đọc '{voice_id}' trong hệ thống.")
                voice_name = resolved
            else:
                voice_name = self._tts._default_voice or "Hải Đăng"

            # Tính toán nhiệt độ cảm xúc (Temperature) từ Stability nếu để default
            if temperature is not None and temperature != "default":
                try:
                    temp = float(temperature)
                    temp = max(0.3, min(1.2, temp))
                except (TypeError, ValueError):
                    temp = 0.8
            else:
                try:
                    st_val = float(stability)
                    # 50% -> 0.80; 80% -> 0.62; 30% -> 0.92
                    temp = max(0.35, min(1.15, 1.10 - (st_val / 100.0) * 0.60))
                except Exception:
                    temp = 0.8

            # Phân tách thành từng câu phụ đề và tạo audio
            sub_cues = split_text_to_subtitles(gen_text, max_chars=max_chars_per_line)
            if not sub_cues:
                sub_cues = [{"text": gen_text, "pause_type": "sentence"}]

            audio_segments: List[np.ndarray] = []
            processed_cues: List[Dict[str, Any]] = []
            current_time = 0.0

            for i, cue_item in enumerate(sub_cues):
                cue_raw_text = cue_item["text"]
                custom_pause = cue_item.get("custom_pause")

                # Nếu là khối chỉ có thẻ nghỉ (ví dụ đặt [nghỉ:Xs] ngay ở đầu bài)
                if not cue_raw_text.strip():
                    if custom_pause and custom_pause > 0:
                        p_samples = int(custom_pause * self.sample_rate)
                        audio_segments.append(np.zeros(p_samples, dtype=np.float32))
                        current_time += custom_pause
                    continue

                # Tạo audio cho phân đoạn câu này
                seg_audio = self._infer_block_smart(
                    cue_raw_text, voice_name, temp,
                    silence_p=silence_p, max_chars=max_chars,
                    auto_normalize=normalize_text,
                    custom_dict=custom_dict,
                )

                if len(seg_audio) == 0:
                    continue

                duration_sec = len(seg_audio) / self.sample_rate
                start_sec = current_time
                end_sec = current_time + duration_sec

                processed_cues.append({
                    "id": len(processed_cues) + 1,
                    "start": start_sec,
                    "end": end_sec,
                    "text": cue_raw_text,
                })

                audio_segments.append(seg_audio)

                # Khoảng lặng giữa các câu: ưu tiên custom_pause nếu có thẻ [nghỉ:Xs]
                if custom_pause is not None:
                    p_sec = custom_pause
                elif i < len(sub_cues) - 1:
                    p_type = cue_item["pause_type"]
                    if p_type == "comma":
                        p_sec = comma_pause
                    elif p_type == "sentence":
                        p_sec = sentence_pause
                    else:
                        p_sec = segment_pause
                else:
                    p_sec = 0.0

                if p_sec > 0:
                    p_samples = int(p_sec * self.sample_rate)
                    if p_samples > 0:
                        audio_segments.append(np.zeros(p_samples, dtype=np.float32))
                    current_time = end_sec + p_sec
                else:
                    current_time = end_sec

            if not audio_segments:
                raise RuntimeError("Mô hình không tạo được audio từ văn bản.")

            audio = np.concatenate(audio_segments) if len(audio_segments) > 1 else audio_segments[0]

            # 1. Điều chỉnh tốc độ đọc (speed) bằng WSOLA miền thời gian (100% tròn tiếng, không bị ồm)
            try:
                target_speed = float(speed)
            except (TypeError, ValueError):
                target_speed = 1.0

            if abs(target_speed - 1.0) >= 0.02:
                try:
                    audio = wsola_time_stretch(audio, rate=target_speed, sr=self.sample_rate)
                    # Cân chỉnh lại mốc thời gian phụ đề theo tỷ lệ tốc độ
                    for c in processed_cues:
                        c["start"] = round(c["start"] / target_speed, 3)
                        c["end"] = round(c["end"] / target_speed, 3)
                except Exception as stretch_err:
                    print(f"⚠️ Không thể co giãn tốc độ: {stretch_err}")

            # 2. Điều chỉnh cao độ (pitch) bằng Resample + WSOLA (không dùng phase vocoder)
            try:
                pitch_steps = float(pitch)
            except (TypeError, ValueError):
                pitch_steps = 0.0

            if abs(pitch_steps) >= 0.25:
                try:
                    audio = clean_pitch_shift(audio, n_steps=pitch_steps, sr=self.sample_rate)
                except Exception as pitch_err:
                    print(f"⚠️ Không thể dịch cao độ: {pitch_err}")

            # 3. Khử tiếng ồm (de-boom), lọc DC offset & Speaker Boost chuẩn phát thanh
            if len(audio) > 0:
                audio = audio - np.mean(audio)
                try:
                    from scipy.signal import butter, sosfilt
                    # Highpass 65Hz khử sub-bass rumble, tiếng rung micro
                    sos = butter(2, 65, btype="highpass", fs=self.sample_rate, output="sos")
                    audio = sosfilt(sos, audio).astype(np.float32)
                except Exception:
                    pass

                target_peak = 0.95 if speaker_boost else 0.90
                max_amp = float(np.max(np.abs(audio)))
                if max_amp > 1e-4:
                    audio = audio * (target_peak / max_amp)

            # Lưu file âm thanh
            self._tts.save(audio, output_file)

            # Lưu file phụ đề .SRT nếu được yêu cầu
            srt_path_saved = None
            if generate_srt and output_srt_path:
                try:
                    srt_file = Path(output_srt_path)
                    srt_file.parent.mkdir(parents=True, exist_ok=True)
                    srt_content = build_srt_content(processed_cues)
                    srt_file.write_text(srt_content, encoding="utf-8")
                    srt_path_saved = str(srt_file)
                except Exception as srt_err:
                    print(f"⚠️ Không thể lưu file .srt: {srt_err}")

        return {
            "audio_path": str(output_file),
            "srt_path": srt_path_saved,
            "duration": round(float(len(audio) / self.sample_rate), 2),
            "cues": processed_cues,
        }
