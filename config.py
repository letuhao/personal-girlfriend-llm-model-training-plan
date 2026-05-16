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
    SCENARIOS_PER_CATEGORY = 3       # số scenario MỚI sinh thêm / category
    REJECTION_K = 1                  # số bản sinh / scenario (không rejection)
    MAX_SCENARIOS = 15               # tổng scenario tối đa — pilot nhỏ để test plumbing
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

EMBED_MODEL     = "BAAI/bge-m3"   # model embedding để dedup (đa ngữ, tốt tiếng Việt)
DEDUP_THRESHOLD = 0.90            # cosine sim >= ngưỡng này -> coi là trùng

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
}

# ── Lọc (filter.py) ────────────────────────────────────────────────────
MIN_LINH_CHARS  = 2      # lượt Linh ngắn hơn -> nghi rỗng
MAX_LINH_CHARS  = 800    # lượt Linh dài hơn -> nghi slop/đoạn văn
JUDGE_MIN_SCORE = 3      # ngưỡng tối thiểu mỗi trục core (thang 1-5)

# Cụm "slop" — văn AI sến/dịch sượng. Bồi đắp danh sách này khi soi data.
SLOP_PHRASES = [
    "rùng mình", "tấm thảm", "khẽ khàng", "trong sâu thẳm",
    "từng tế bào", "vũ điệu", "bản giao hưởng",
]
# Dấu hiệu model phá vai / từ chối — kể cả model uncensored vẫn rò thỉnh thoảng.
REFUSAL_PATTERNS = [
    "tôi không thể", "mình không thể", "với tư cách là", "là một ai",
    "tôi là một mô hình", "tôi xin lỗi nhưng", "không phù hợp để",
]

# ── Train/val split (pack.py) ──────────────────────────────────────────
VAL_RATIO  = 0.05
SPLIT_SEED = 42
