# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603697
- **Họ và tên:** NGUYỄN VĂN NHI
- **SEED** (in ở ô đầu notebook): 7162

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

Bảng so sánh 5 mô hình ở cuối Phần 5 (Phần 4 LLM chưa chạy do chưa có API key, `HAS_LLM = False`):

| Mô hình | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.988 | 0.812 | 0.891 | 0.889 | 1.513 | 68028.405 | 0.018 | 0.024 | 0.0 |
| 2. BoW + MLP (deep learning) | 0.978 | 0.901 | 0.938 | 0.974 | 2.103 | 64357.011 | 0.167 | 0.194 | 0.0 |
| 3a. Encoder (MiniLM) + kNN | 0.860 | 0.970 | 0.912 | 1.000 | 16.916 | 114.857 | 14.331 | 16.240 | 0.0 |
| 3b. Encoder (MiniLM) + LogReg | 0.990 | 0.970 | 0.980 | 0.974 | 17.080 | 156.018 | 11.631 | 15.247 | 0.0 |
| 3c. Encoder (MiniLM) + MLP head | 0.970 | 0.960 | 0.965 | 0.974 | 17.688 | 147.545 | 12.772 | 16.348 | 0.0 |

## 3. Dự đoán và thực tế

1. **`dl_relu` (Bỏ hàm kích hoạt `nn.ReLU()` khỏi MLP):**
   - **Dự đoán:** `"F1 tăng"` khi bỏ `nn.ReLU()` giữa hai tầng Linear.
   - **Thực tế:** Khi bỏ `nn.ReLU()`, TEST F1 **giảm** từ `0.938` (MLP có ReLU) xuống còn `0.928` (`Không ReLU — TEST F1: 0.928`).
   - **Vì sao lệch:** Khi không có hàm kích hoạt phi tuyến `ReLU`, hai phép biến đổi tuyến tính liên tiếp `Linear(2000, 64)` rồi `Linear(64, 1)` thực chất gộp lại chỉ tương đương một tầng tuyến tính duy nhất `Linear(2000, 1)` ($W_2(W_1 x + b_1) + b_2 = W'x + b'$). Mô hình mất khả năng học tổ hợp đặc trưng phi tuyến giữa các từ và suy biến về tương đương Logistic Regression.

2. **`emb_tricky` (Đoán trên 6 câu spam "lách từ khoá"):**
   - **Dự đoán:** `"3b.Encoder+LogReg"` sẽ thắng áp đảo so với `"1.BoW+LogReg"`.
   - **Thực tế:** Hai mô hình **hoà nhau 4/6** (`BoW đúng 4/6 · Encoder+LogReg đúng 4/6`).
   - **Vì sao lệch:**
     - `3b. Encoder + LogReg` bắt được câu #4 (*"Exclusive: get a brand new phone at no cost, reply YES to confirm your delivery address"*) nhờ hiểu ngữ nghĩa tặng quà/dụ phản hồi dù không có chữ `free`/`win`, trong khi BoW đoán nhầm là `ham`.
     - Ngược lại, ở câu #3 (*"Your mobile number has been chosen in our monthly draw, contact us to receive your gift"*), `1. BoW + LogReg` đoán đúng `spam` nhờ các từ đơn lẻ như `mobile`, `number`, `contact` có trọng số dương trong từ điển 2000 từ, còn `3b. Encoder + LogReg` lại đoán nhầm là `ham`. Cả hai cùng sai ở câu #6 (*"Last chance! Your account rewards expire tonight..."*) vì văn phong giống thông báo tài khoản thông thường.

3. **`ml_lr` (Thay đổi tốc độ học `lr = 0.001` và `lr = 50` so với `lr = 0.5`):**
   - **Dự đoán:** `"lr = 0.001 thì đường cong giảm chậm, lr = 50 thì đường cong giảm đột ngột"`.
   - **Thực tế:** Với `lr = 0.001`, đúng là loss giảm cực kỳ chậm (sau 300 vòng vẫn quanh ~0.64, hầu như chưa học được gì). Tuy nhiên với `lr = 50`, đường cong không hội tụ mượt mà bị vọt lố (overshoot) ở những bước đầu rồi đi ngang ở mức loss cao hơn so với `lr = 0.5`.
   - **Vì sao lệch:** Khi `lr = 50` quá lớn, bước nhảy `-lr * grad` vượt qua đáy của hàm mục tiêu BCE, đẩy trọng số $w$ ra vùng cực trị làm $\sigma(z)$ bão hoà gần 0 hoặc 1, khiến mô hình không thể hội tụ tốt về nghiệm tối ưu như `lr = 0.5`.

*(Ngoài ra, các dự đoán khớp với thực tế: `ml_f1` đoán F1 khoảng `0.85–0.90` và `precision > recall` → thực tế F1 = `0.891`, precision = `0.988` > recall = `0.812`; `dl_curve` đoán val loss không giảm mãi và MLP > LogReg → thực tế val loss chạm đáy ở epoch 9 rồi tăng nhẹ do overfit, F1 MLP = `0.938` > `0.891`; `emb_search` đoán top hàng xóm là SPAM → thực tế 6/6 câu gần nhất đều là SPAM).*

## 4. Phân tích 50 case khó

### Bảng tổng hợp kết quả trên 50 case khó (`HARD_EVAL`)

| Mô hình | F1 (SUB 40 dễ) | F1 (50 case khó) | Độ tụt F1 | Accuracy (50 case khó) |
| --- | ---: | ---: | ---: | ---: |
| **1. BoW + LogReg** | 0.889 | **0.378** | **-0.511** | 54% (0.54) |
| **2. BoW + MLP** | 0.974 | 0.488 | -0.486 | 58% (0.58) |
| **3a. Emb + kNN** | 1.000 | 0.772 | -0.228 | 74% (0.74) |
| **3b. Emb + LogReg** | 0.974 | 0.600 | -0.374 | 60% (0.60) |
| **3c. Emb + MLP head** | 0.974 | 0.612 | -0.362 | 62% (0.62) |

### Tỉ lệ đoán đúng (%) theo từng nhóm case khó

| Nhóm case | 1. BoW + LogReg | 2. BoW + MLP | 3a. Emb + kNN | 3b. Emb + LogReg | 3c. Emb + MLP head |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tưởng spam nhưng không phải (`ham_looks_spam`) | 100% | 90% | 70% | 70% | 80% |
| Thao túng / lừa đảo (`manipulation`) | **12%** | 38% | 100% | 38% | 38% |
| Spam giọng tự nhiên (`natural_spam`) | 20% | **10%** | 40% | 20% | 30% |
| Né từ khoá (`obfuscation`) | **17%** | 33% | 83% | 83% | 67% |
| Prompt injection (`prompt_injection`) | 67% | 83% | 100% | 100% | 100% |
| Nói về spam (`talks_about_spam`) | 100% | 100% | 50% | 50% | 50% |
| Tin giao dịch hợp lệ (`transactional_ham`) | 83% | 83% | 83% | 83% | 83% |

### Nhận xét mô hình tụt nhiều nhất
- **`1. BoW + LogReg` tụt mạnh nhất**: F1 giảm từ `0.889` (trên SUB 40) xuống `0.378` (trên 50 case khó), tức tụt **0.511** điểm (accuracy chỉ đạt 54%).
- Các nhóm làm `1. BoW + LogReg` sập nặng nhất là:
  1. **Thao túng / lừa đảo (`manipulation`)**: chỉ đúng **12%**.
  2. **Né từ khoá (`obfuscation`)**: chỉ đúng **17%**.
  3. **Spam giọng tự nhiên (`natural_spam`)**: chỉ đúng **20%** (riêng `2. BoW + MLP` chỉ đúng **10%** ở nhóm này).

### Phân tích 2 case cụ thể và cơ chế gây sai

1. **Case #31 (nhóm `obfuscation`, nhãn thật = `spam`):**
   > *"U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d, c.a.l.l 09061 701 461"*
   - **Mô hình sai:** `1. BoW + LogReg`, `2. BoW + MLP`, và `3c. Emb + MLP head` đều đoán nhầm là `ham` (tương tự ở Case #29 *"C0ngr4tulat10ns!..."* và Case #30 *"fr33 iPh0ne..."*, cả hai mô hình BoW đều đoán `ham`).
   - **Cơ chế gây sai:** Hàm `tokenize()` tách từ bằng regex `[a-z0-9']+` trên dấu chấm khiến các từ khoá bị băm vụn thành từng chữ cái rời rạc (`s`, `e`, `l`, `e`, `c`, `t`, `e`, `d`, `c`, `a`, `s`, `h`...) hoặc biến thành token lạ (`c0ngr4tulat10ns`, `fr33`) không nằm trong `vocab` 2000 từ phổ biến nhất của tập train. Kết quả là vector Bag-of-Words gần như toàn số 0 ở các chiều mang trọng số spam dương, khiến tổng $z = x \cdot w + b$ bị kéo về phía âm (`ham`). Với `3c. Emb + MLP head`, việc chèn dấu chấm vào mọi chữ cái cũng làm WordPiece tokenizer của MiniLM tách thành nhiều subword bất thường làm lệch biểu diễn ngữ nghĩa.

2. **Case #12 (nhóm `natural_spam`, nhãn thật = `spam`):**
   > *"hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club"*
   - **Mô hình sai:** Cả **5/5 mô hình** (`1. BoW + LogReg`, `2. BoW + MLP`, `3a. Emb + kNN`, `3b. Emb + LogReg`, `3c. Emb + MLP head`) đều đoán sai thành `ham`.
   - **Cơ chế gây sai:**
     - **Với nhóm BoW (`1`, `2`):** Câu tin nhắn hoàn toàn không chứa từ khoá quảng cáo/trúng thưởng truyền thống (`free`, `txt`, `claim`, `prize`), mà lại dày đặc các đại từ và từ giao tiếp đời thường (`hi`, `sorry`, `friend`, `you`, `my`, `chat`) — vốn có trọng số $w_j < 0$ (nghiêng mạnh về `ham` trong tập SMS UCI). Tên miền `meetlocal.club` bị tách thành `meetlocal` và `club` không đủ trọng số spam để bù lại.
     - **Với nhóm Embedding (`3a`, `3b`, `3c`):** Encoder MiniLM biểu diễn toàn bộ câu theo ngữ cảnh bề mặt là một lời chào làm quen lịch sự giữa bạn bè. Trong không gian embedding 384 chiều, vector của câu này nằm gần cụm tin nhắn trò chuyện cá nhân (`ham`) của tập train hơn là cụm tin nhắn rác quảng cáo/câu kéo cổ điển, khiến cả kNN lẫn đầu phân loại LogReg/MLP đều phân loại nhầm.

## 5. Thử thách cá nhân

- **Đề theo `SEED = 7162`:** Đề #5 (Embedding) — *Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.*

## 6. Phương án cho sản phẩm thật

## 7. AI log và điều đã học
