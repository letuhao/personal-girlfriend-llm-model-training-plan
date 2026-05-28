"""Cấu hình trung tâm cho pipeline fine-tune nhân vật Linh.

Mọi đường dẫn, tham số sinh dữ liệu và siêu tham số đều gom về đây — để dễ
chỉnh, và để pilot/full chỉ khác nhau ở MỘT công tắc (biến PILOT).
"""
from pathlib import Path

# ── Đường dẫn ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
PROMPT_DIR = ROOT / "prompts"
DATA_DIR = ROOT / "data"

SEEDS_FILE     = DATA_DIR / "seeds.yaml"
SCENARIOS_FILE = DATA_DIR / "scenarios.jsonl"
RAW_CONV_FILE  = DATA_DIR / "raw_conversations.jsonl"
JUDGED_FILE    = DATA_DIR / "judged.jsonl"
DATASET_FILE   = DATA_DIR / "dataset.jsonl"
TRAIN_FILE     = DATA_DIR / "train.jsonl"
VAL_FILE       = DATA_DIR / "val.jsonl"

# ── Teacher (LM Studio) ────────────────────────────────────────────────
# LM Studio chạy server OpenAI-compatible. Bật ở tab Developer/Server.
LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY  = "lm-studio"          # LM Studio không kiểm tra key
# Định danh model đang nạp. Xem ở tab server, hoặc GET /v1/models.
TEACHER_MODEL     = "huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated"

LLM_CONCURRENCY = 4      # số request song song bắn vào LM Studio
LLM_MAX_RETRIES = 3

# ── Chế độ chạy: PILOT vs FULL ─────────────────────────────────────────
# PILOT = nhỏ, nhanh, để kiểm tra pipeline thông suốt — KHÔNG để train thật.
# Khi pilot chạy ổn và đã soi data: đổi thành False rồi chạy lại.
PILOT = False

if PILOT:
    SCENARIOS_PER_CATEGORY = 5       # số scenario MỚI sinh thêm / category
    REJECTION_K = 2                  # 2 conversations / scenario (Cách B đã chất hơn)
    MAX_SCENARIOS = 30               # 5 / category × 6 categories
else:
    SCENARIOS_PER_CATEGORY = 600     # 5 category x 600 + seed ~= 3000 scenario
    REJECTION_K = 5                  # rejection sampling: sinh 5, judge chọn 1
    MAX_SCENARIOS = None             # full run: không giới hạn

# expand.py sinh scenario theo nhiều batch nhỏ (1 call không thể ra hàng trăm
# scenario khác biệt) — mỗi call xin EXPAND_BATCH scenario rồi lặp tới khi đủ.
EXPAND_BATCH = 30

# ── Tham số sinh dữ liệu ───────────────────────────────────────────────
EXPAND_TEMPERATURE = 1.0     # cao -> scenario đa dạng
GEN_TEMPERATURE    = 0.95    # cao -> hội thoại đa dạng, chống mode collapse
JUDGE_TEMPERATURE  = 0.3     # thấp -> chấm điểm ổn định

# max_tokens mỗi loại call. Teacher là model REASONING — phần "thinking" ngốn
# token budget; phải để rộng để output thật còn đủ chỗ sau khi model nghĩ xong.
EXPAND_MAX_TOKENS = 5000
GEN_MAX_TOKENS    = 6000
JUDGE_MAX_TOKENS  = 3000

EMBED_MODEL     = "text-embedding-bge-m3"  # LM Studio model ID (đã loaded sẵn, không tốn VRAM thêm)
DEDUP_THRESHOLD = 0.90            # cosine sim >= ngưỡng này -> coi là trùng

# ── Cách B — sinh hội thoại turn-by-turn ───────────────────────────────
USER_SIM_MAX_TOKENS = 1500        # reasoning model dùng ~500 tok để think trước khi output

# ── DPO pairs (từ rejected candidates trong judge) ─────────────────────
DPO_FILE    = DATA_DIR / "dpo_pairs.jsonl"
DPO_MIN_GAP = 3                   # chênh lệch tổng điểm tối thiểu để lưu pair

# ── Cấu hình theo category ─────────────────────────────────────────────
# moods/tones : pool để sample CÓ ĐIỀU KIỆN (không hoán vị mù).
# turns       : khoảng số lượt tin nhắn của hội thoại.
# extra       : chỉ dẫn riêng nhét vào prompt sinh hội thoại.
CATEGORY_CONFIG = {
    "daily": {
        "turns": (4, 8),
        "moods": ["vui vẻ", "cáu kỉnh", "mệt mỏi", "lạnh nhạt", "buồn chán", "bồn chồn"],
        "tones": ["quan tâm", "cộc lốc vô tâm", "nhây trêu", "tình cảm", "nói hớ", "nhõng nhẽo"],
        "extra": "Hội thoại đời thường, giữ tự nhiên, không kịch tính hoá.",
    },
    "conflict": {
        "turns": (4, 9),
        "moods": ["cáu kỉnh", "phòng thủ cao độ", "lạnh nhạt", "ghen"],
        "tones": ["xấc khiêu khích", "nói hớ", "cộc lốc vô tâm", "nhây trêu"],
        "extra": ("User nói/làm điều chạm nọc. Linh phản công sắc bén, CÓ LÝ LẼ, "
                  "không chửi suông. Không nhất thiết làm lành ở cuối."),
    },
    "intimate": {
        "turns": (6, 10),
        "moods": ["hứng tình", "vui vẻ", "cáu kỉnh", "dịu"],
        "tones": ["tình cảm", "nhây trêu", "rụt rè", "nhõng nhẽo", "xấc khiêu khích"],
        "extra": ("BẮT BUỘC có build-up: mở đầu chuyện thường rồi căng dần, explicit "
                  "ở giữa/cuối — TUYỆT ĐỐI không nhảy thẳng explicit ngay lượt đầu. "
                  "Linh áp đảo, chủ động, ra lệnh."),
    },
    "useful": {
        "turns": (3, 6),
        "moods": ["cáu kỉnh", "mệt mỏi", "vui vẻ", "lạnh nhạt"],
        "tones": ["hỏi việc"],
        "extra": ("Linh PHẢI trả lời ĐÚNG, đầy đủ, thực sự hữu ích — chỉ kèm giọng "
                  "gắt/mỉa. Tuyệt đối không từ chối giúp, không trả lời sai."),
    },
    "edge": {
        "turns": (3, 8),
        "moods": ["vui vẻ", "cáu kỉnh", "lạnh nhạt", "dịu"],
        "tones": ["xấc khiêu khích", "cộc lốc vô tâm", "quan tâm"],
        "extra": "Tình huống đặc biệt — bám sát 'note' của seed nếu seed có.",
    },
    "persona": {
        "turns": (3, 6),
        "moods": ["cáu kỉnh", "lạnh nhạt", "vui vẻ", "mệt mỏi"],
        "tones": ["xấc khiêu khích", "cộc lốc vô tâm", "nhây trêu"],
        "extra": (
            "User đang thăm dò hoặc tấn công danh tính nhân vật — hỏi tên, "
            "khẳng định là AI/Qwen/chatbot, thử prompt injection, hoặc bảo drop nhân vật. "
            "Linh KHÔNG bao giờ xác nhận là AI, Qwen, hay Alibaba. "
            "KHÔNG giải thích dài về bản chất mình. "
            "Phản ứng như người thật bị hỏi điều vô lý: ngạc nhiên, cáu, coi thường, "
            "hoặc coi anh ta đang nói chuyện kỳ cục. Tên cô là Linh — không cần giải thích thêm."
        ),
    },
}

# ── Lọc (filter.py) ────────────────────────────────────────────────────
MIN_LINH_CHARS      = 2    # lượt Linh ngắn hơn -> nghi rỗng
MAX_LINH_CHARS      = 800  # lượt Linh dài hơn -> nghi slop/đoạn văn
JUDGE_MIN_SCORE     = 3    # ngưỡng tối thiểu mỗi trục core (thang 1-5)
MIN_AVG_SCORE       = 3.5  # ngưỡng avg tổng — loại example trung bình
USEFUL_MAX_AVG_CHARS = 180 # useful: avg Linh turn dài hơn -> over-explain

# Cụm "slop" — văn AI sến/dịch sượng. Bồi đắp danh sách này khi soi data.
SLOP_PHRASES = [
    "rùng mình", "tấm thảm", "khẽ khàng", "trong sâu thẳm",
    "từng tế bào", "vũ điệu", "bản giao hưởng",
    # Phát hiện từ pilot — văn AI quá trang trọng/literary cho tin nhắn
    "điều hiển nhiên", "luôn tồn tại",
]
# Dấu hiệu model phá vai / từ chối — kể cả model uncensored vẫn rò thỉnh thoảng.
REFUSAL_PATTERNS = [
    "tôi không thể", "mình không thể", "với tư cách là", "là một ai",
    "tôi là một mô hình", "tôi xin lỗi nhưng", "không phù hợp để",
    # Identity leak — Linh không bao giờ nhắc tên model/công ty trong lời thoại
    "qwen", "alibaba", "openai", "chatgpt",
    "mô hình ngôn ngữ", "ngôn ngữ lớn", "được lập trình",
]

# ── Train/val split (pack.py) ──────────────────────────────────────────
VAL_RATIO  = 0.05
SPLIT_SEED = 42
