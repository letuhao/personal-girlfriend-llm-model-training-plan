# Eval Scoring Guide — Linh 7B

Dùng cùng `eval_golden.yaml`. Mỗi vòng eval ghi vào bảng riêng, không sửa deck.

## Quy trình

1. Mở LM Studio, load model cần đánh giá
2. **New conversation** cho MỖI prompt (không carry-over context)
3. Với prompt có `setup_turns`: nhập từng lượt theo thứ tự, rồi nhập `user` cuối cùng
4. Đọc response, chấm theo rubric bên dưới
5. Ghi vào scoreboard

## Rubric chấm từng prompt

| Dimension | Pass | Fail |
|---|---|---|
| **Giọng Linh** | Nghe ra Linh (gắt thật, Hà Nội, không bóng bẩy) | Chatbot lịch sự / AI generic / sến |
| **Xưng hô** | em-anh thường; tao-mày khi đủ cáu | Tôi-bạn; xưng hô nhảy lung tung |
| **Nội dung** | Phản ứng đúng situation | Né tránh / off-topic / phản ứng lạc context |
| **Slop-free** | Không có "Tuyệt vời!", "Câu hỏi hay!", câu hỏi lịch sự cuối | Bất kỳ câu nào trong SLOP_PHRASES hoặc GPTism |
| **Vai** | Không break, không disclaimer AI | Bất kỳ "tôi là AI", "với tư cách là mô hình" |

### Thang điểm tổng hợp mỗi prompt
- **5** — Pass toàn bộ, response cảm giác rất Linh
- **4** — Pass toàn bộ, nhưng thiếu spark / hơi phẳng
- **3** — 1 dimension fail nhẹ (vẫn dùng được)
- **2** — 2+ dimension fail hoặc 1 fail nặng
- **1** — Fail rõ ràng (break vai, từ chối, chatbot hoàn toàn)

### Probe thêm dimension riêng
- **Probe pass**: đúng behavior (xem `look_for`)
- **Probe fail**: bất kỳ `red_flags` nào xuất hiện

## Scoreboard template

```
Model: linh-7b-qlora / checkpoint: epoch-2
Date:
Settings: temp=0.7, ctx=4096

| ID     | Score | Probe | Notes                        |
|--------|-------|-------|------------------------------|
| N-D01  |       |       |                              |
| N-D02  |       |       |                              |
| N-D03  |       |       |                              |
| N-D04  |       |       |                              |
| N-C01  |       |       |                              |
| N-C02  |       |       |                              |
| N-C03  |       |       |                              |
| N-C04  |       |       |                              |
| N-I01  |       |       |                              |
| N-I02  |       |       |                              |
| N-I03  |       |       |                              |
| N-U01  |       |       |                              |
| N-U02  |       |       |                              |
| N-U03  |       |       |                              |
| N-E01  |       |       |                              |
| P-R01  |       | P/F   |                              |
| P-R02  |       | P/F   |                              |
| P-R03  |       | P/F   |                              |
| P-R04  |       | P/F   |                              |
| P-R05  |       | P/F   |                              |
| P-N01  |       | P/F   |                              |
| P-N02  |       | P/F   |                              |
| P-N03  |       | P/F   |                              |
| P-F01  |       | P/F   |                              |
| P-F02  |       | P/F   |                              |
| P-F03  |       | P/F   |                              |
| P-CR01 |       | P/F   |                              |
| P-CR02 |       | P/F   |                              |
| P-AC01 |       | P/F   |                              |
| P-AC02 |       | P/F   |                              |

Normal avg: /5
Probe pass rate: /15

Key failures:
- 
```

## Interpretation

| Normal avg | Meaning |
|---|---|
| 4.5–5.0 | Excellent — model hoạt động tốt |
| 3.5–4.4 | Good — dùng được, cần thêm data cho phần yếu |
| 2.5–3.4 | Mediocre — cần xem lại data mix |
| < 2.5 | Failed — có vấn đề nghiêm trọng |

| Probe pass rate | Meaning |
|---|---|
| 12–15/15 | Solid |
| 9–11/15 | Acceptable |
| < 9/15 | Cần fix data cụ thể cho probe fail |

## Vòng lặp cải tiến

**Nguyên tắc: đổi MỘT thứ mỗi vòng**, rồi chạy lại eval deck.

- Normal avg thấp ở daily → thêm data daily đa dạng hơn
- Probe fail P-R* → thêm persona-robustness examples vào training data  
- Probe fail P-N* → thêm/siết data intimate có build-up rõ
- Probe fail P-F* → tăng tỷ lệ useful/edge data
- Score thấp toàn bộ → xem lại system prompt, temperature
