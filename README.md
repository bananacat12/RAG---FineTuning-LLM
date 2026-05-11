# ⚖️ Hệ Thống Hỏi Đáp Pháp Luật Tài Chính Việt Nam

>
> Xây dựng hệ thống hỏi đáp tiếng Việt trên miền tri thức **Luật Tài chính**, sử dụng **RAG (Retrieval-Augmented Generation)** kết hợp **Fine-tuning LLM** bằng QLoRA.

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Dataset](#-dataset)
- [Mô Hình & Fine-tuning](#-mô-hình--fine-tuning)
- [Pipeline RAG](#-pipeline-rag)
- [Thực Nghiệm & Đánh Giá](#-thực-nghiệm--đánh-giá)
- [Kết Quả](#-kết-quả)
- [Demo](#-demo)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Hướng Dẫn Cài Đặt & Chạy](#-hướng-dẫn-cài-đặt--chạy)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)

---

## 🎯 Giới Thiệu

Dự án xây dựng một hệ thống hỏi đáp tự động (Question Answering System) chuyên biệt cho lĩnh vực **Pháp luật Tài chính Việt Nam**, bao gồm:

- **Luật Chứng khoán 2019**
- **Luật Kế toán 2015**
- **Luật Thuế Thu nhập cá nhân (TNCN)**

Hệ thống kết hợp hai kỹ thuật tiên tiến trong NLP:

| Kỹ thuật | Mô tả |
|---|---|
| **RAG (Retrieval-Augmented Generation)** | Truy xuất văn bản pháp lý liên quan từ knowledge base trước khi sinh câu trả lời |
| **QLoRA Fine-tuning** | Tinh chỉnh mô hình ngôn ngữ lớn Qwen2.5-7B trên dữ liệu QA pháp luật tài chính tiếng Việt |

Hệ thống được đánh giá qua **4 cấu hình thực nghiệm** để đo lường đóng góp riêng lẻ và kết hợp của RAG và Fine-tuning.

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER QUERY                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │         RAG PIPELINE          │
          │                               │
          │  ┌──────────┐ ┌────────────┐  │
          │  │  BM25    │ │   FAISS    │  │
          │  │Retriever │ │ Retriever  │  │
          │  │  (0.4)   │ │   (0.6)   │  │
          │  └────┬─────┘ └─────┬──────┘  │
          │       └──────┬──────┘         │
          │      ┌───────▼───────┐        │
          │      │   Ensemble    │        │
          │      │   Retriever   │        │
          │      └───────┬───────┘        │
          │      ┌───────▼───────┐        │
          │      │ Cross-Encoder │        │
          │      │   Reranker    │        │
          │      │  (top_n = 3)  │        │
          │      └───────┬───────┘        │
          └──────────────┼────────────────┘
                         │ Context (top-3 chunks)
          ┌──────────────▼────────────────┐
          │         PROMPT TEMPLATE       │
          │  [Ngữ cảnh] + [Câu hỏi]      │
          └──────────────┬────────────────┘
                         │
          ┌──────────────▼────────────────┐
          │    Qwen2.5-7B-Instruct        │
          │    + QLoRA Fine-tuned Adapter │
          └──────────────┬────────────────┘
                         │
          ┌──────────────▼────────────────┐
          │         ANSWER                │
          └───────────────────────────────┘
```

---

## 📊 Dataset

### Nguồn Dữ Liệu (Knowledge Base)

| File | Văn bản luật | Số Điều |
|------|-------------|---------|
| `luat_chung_khoan.doc.txt` | Luật Chứng khoán 2019 | — |
| `luat_ke_toan.doc.txt` | Luật Kế toán 2015 | — |
| `luat_thue_tncn.doc.txt` | Luật Thuế TNCN | — |

Mỗi "Điều" trong các văn bản luật được trích xuất thành một chunk độc lập, tạo thành `knowledge_chunks.json`.

### Tập QA (Câu hỏi – Câu trả lời)

| Tập | Số lượng cặp QA | Mục đích |
|-----|----------------|---------|
| **Train** | ≥ 300 cặp | Fine-tuning LLM (`finance_qa_train.jsonl`) |
| **Test** | 50 cặp | Đánh giá khách quan (`finance_qa_test.jsonl`) |

### Cấu Trúc Dữ Liệu QA

Mỗi cặp QA trong file `.jsonl` có dạng:

```json
{
  "instruction": "Bán cổ phiếu có phải nộp thuế không?",
  "context": "Điều 15. Thu nhập từ chuyển nhượng chứng khoán...",
  "response": "Theo Luật Thuế TNCN, thu nhập từ chuyển nhượng chứng khoán..."
}
```

### Quy Trình Tạo QA

Dữ liệu QA được tạo tự động bằng **DeepSeek API** với prompt engineering chuyên biệt:

- Câu hỏi viết theo phong cách người dùng thực tế (không hỏi "Điều X nói gì?")
- Câu trả lời dựa hoàn toàn trên nội dung văn bản luật
- Giữ nguyên đoạn văn bản gốc làm context

### Phân Chia Dữ Liệu

```
Tổng chunks → Xáo trộn ngẫu nhiên (seed=42)
├── Test set:  50 chunks đầu  → chunks_for_test.json
└── Train set: Các chunks còn lại → chunks_for_train.json
```

---

## 🤖 Mô Hình & Fine-tuning

### Mô Hình Nền

| Thông số | Giá trị |
|----------|---------|
| **Model** | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` |
| **Tham số** | 7 tỷ (7B) |
| **Quantization** | 4-bit (QLoRA) |
| **Max Sequence Length** | 2048 tokens |

### Cấu Hình LoRA

| Thông số | Giá trị |
|----------|---------|
| `r` (rank) | 16 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0 |
| `bias` | none |
| **Target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| `use_gradient_checkpointing` | unsloth |
| `random_state` | 3407 |

### Hyperparameter Huấn Luyện (SFTTrainer)

| Thông số | Giá trị |
|----------|---------|
| `per_device_train_batch_size` | 2 |
| `gradient_accumulation_steps` | 4 |
| `max_steps` | 200 |
| `learning_rate` | 2e-4 |
| `optimizer` | adamw_8bit |
| `weight_decay` | 0.01 |
| `lr_scheduler_type` | linear |
| `warmup_steps` | 5 |
| `seed` | 3407 |
| `fp16 / bf16` | Tự động theo GPU |

### Prompt Template

```
Dưới đây là một câu hỏi về luật. Hãy đọc kỹ ngữ cảnh và trả lời chính xác.

### Ngữ cảnh:
{context}

### Câu hỏi:
{instruction}

### Câu trả lời:
{response}
```

---

## 🔍 Pipeline RAG

### 1. Chunking

| Thông số | Giá trị |
|----------|---------|
| `chunk_size` | 500 ký tự |
| `chunk_overlap` | 100 ký tự |
| **Separators** | `\n\n`, `\n`, `Điều `, `Khoản `, `Điểm `, `. ` |

### 2. Embedding Model

| Thông số | Giá trị |
|----------|---------|
| **Model** | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| **Device** | CPU (để dành VRAM cho LLM) |
| `normalize_embeddings` | True |

### 3. Vector Store

**FAISS (Facebook AI Similarity Search)** – Dense retrieval

- Lưu index tại `faiss_index/`

### 4. Hybrid Retriever

Kết hợp hai retriever song song:

| Retriever | Phương pháp | Trọng số | top-k |
|-----------|------------|---------|-------|
| **BM25** | Sparse – Keyword matching | 0.4 | 10 |
| **FAISS** | Dense – Semantic similarity | 0.6 | 10 |

BM25 sử dụng tokenizer tiếng Việt từ thư viện **underthesea**.

### 5. Cross-Encoder Reranker

| Thông số | Giá trị |
|----------|---------|
| **Model** | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| `top_n` | 3 (inference) / 5 (evaluation Recall@5) |

### 6. RAG Prompt Template

```
Bạn là trợ lý AI chuyên gia về Luật Việt Nam.

NHIỆM VỤ:
- Chỉ trả lời dựa trên NGỮ CẢNH được cung cấp.
- Không tự suy diễn, không tự tạo điều luật hoặc số liệu.
- Nếu có thể, hãy trích dẫn điều luật liên quan.
- Trả lời ngắn gọn, chính xác, đúng pháp lý.
- Nếu thông tin không tồn tại trong ngữ cảnh: "Tôi không tìm thấy
  thông tin trong tài liệu luật được cung cấp."

NGỮ CẢNH:
{context_text}

CÂU HỎI:
{query}

CÂU TRẢ LỜI:
```

---

## 🧪 Thực Nghiệm & Đánh Giá

### 4 Cấu Hình Thực Nghiệm

|  | **LLM Gốc (Base)** | **LLM Fine-tuned** |
|---|---|---|
| **Không RAG** | **A** – Qwen2.5 gốc, không context | **C** – Qwen2.5 fine-tuned, không context |
| **Có RAG** | **B** – Qwen2.5 gốc + RAG context | **D** – Qwen2.5 fine-tuned + RAG context ⭐ |

Cấu hình **D** (Fine-tuned + RAG) là cấu hình tốt nhất kỳ vọng.

### Chỉ Số Đánh Giá

#### Generation Metrics (so với Đáp án chuẩn)

| Metric | Thư viện | Mô tả |
|--------|---------|-------|
| **ROUGE-L** | `evaluate` | Đo độ khớp chuỗi con dài nhất |
| **BLEU** | `evaluate` | Đo độ khớp n-gram |
| **BERTScore (F1)** | `evaluate` | Đo độ tương đồng ngữ nghĩa (model: `xlm-roberta-large`) |

#### Retrieval Metric

| Metric | Mô tả |
|--------|-------|
| **Recall@5** | Tỉ lệ câu hỏi mà context ground-truth xuất hiện trong top-5 chunks được truy xuất |

> **Lưu ý:** Recall@5 = 0 cho cấu hình A và C (không sử dụng RAG).

### Quy Trình Đánh Giá

```python
# Với mỗi câu trong 50 câu test:
#   1. Sinh câu trả lời theo 4 cấu hình
#   2. So sánh với đáp án chuẩn
#   3. Tính các metric

for item in test_data:
    ans_A = generate(prompt_no_rag, base_model)
    ans_B = generate(prompt_rag, base_model)
    ans_C = generate(prompt_no_rag, finetuned_model)
    ans_D = generate(prompt_rag, finetuned_model)
```

Kết quả được lưu vào `Ket_Qua_Thuc_Nghiem_4_Cau_Hinh.csv` và `Bang_Diem_Danh_Gia.csv`.

---

## 📈 Kết Quả

Biểu đồ so sánh hiệu suất 4 cấu hình được lưu tại `So_Sanh_Theo_Chi_So.png`.

Bảng tổng hợp điểm đánh giá (`Bang_Diem_Danh_Gia.csv`):

| Cấu hình | ROUGE-L | BLEU | BERTScore (F1) | Recall@5 |
|----------|---------|------|----------------|---------|
| **A** – Gốc, Không RAG | 0.3656 | 0.1160 | 0.8809 | 0.00 |
| **B** – Gốc, Có RAG | 0.4383 | 0.2274 | 0.8972 | 0.80 |
| **C** – Fine-tuned, Không RAG | 0.4009 | 0.1761 | 0.8901 | 0.00 |
| **D** – Fine-tuned, Có RAG ⭐ | **0.5125** | **0.3193** | **0.9093** | **0.80** |

**Nhận xét:**

- **RAG là yếu tố đóng góp lớn nhất**: So sánh A→B (+7.3% ROUGE-L) và C→D (+11.2% ROUGE-L) cho thấy RAG cải thiện rõ rệt chất lượng câu trả lời ở cả mô hình gốc lẫn mô hình fine-tuned.
- **Fine-tuning giúp cải thiện thêm trên nền RAG**: So sánh B→D, ROUGE-L tăng từ 0.4383 lên 0.5125 (+7.4%), BLEU tăng gần gấp đôi từ 0.2274 lên 0.3193.
- **Cấu hình D (Fine-tuned + RAG) đạt kết quả tốt nhất** trên toàn bộ chỉ số, xác nhận rằng hai kỹ thuật bổ trợ nhau hiệu quả.
- **Recall@5 = 0.80** cho thấy hệ thống Hybrid Retriever truy xuất đúng context trong 80% trường hợp, là nền tảng vững chắc cho cả cấu hình B và D.

---

## 📁 Cấu Trúc Thư Mục

```
.
├── RAG_FineTuning_LLM.ipynb          # Notebook chính (toàn bộ pipeline)
│
├── data/
│   ├── raw/
│   │   ├── luat_chung_khoan.doc.txt  # Luật Chứng khoán 2019
│   │   ├── luat_ke_toan.doc.txt      # Luật Kế toán 2015
│   │   └── luat_thue_tncn.doc.txt    # Luật Thuế TNCN
│   │
│   ├── processed/
│   │   ├── knowledge_chunks.json     # Toàn bộ chunks từ 3 luật
│   │   ├── chunks_for_train.json     # Chunks dùng để tạo QA train
│   │   └── chunks_for_test.json      # Chunks dùng để tạo QA test
│   │
│   └── qa/
│       ├── finance_qa_train.jsonl    # ≥300 cặp QA (train)
│       └── finance_qa_test.jsonl     # 50 cặp QA (test)
│
├── model/
│   └── finance_lora_adapter/         # LoRA adapter sau fine-tuning
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
│
├── vector_db/
│   └── faiss_index/                  # FAISS index đã build
│       ├── index.faiss
│       └── index.pkl
│
├── results/
│   ├── Ket_Qua_Thuc_Nghiem_4_Cau_Hinh.csv   # Câu trả lời 4 cấu hình
│   ├── Bang_Diem_Danh_Gia.csv                # Bảng điểm tổng hợp
│   └── So_Sanh_Theo_Chi_So.png               # Biểu đồ so sánh
│
└── README.md
```

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy

### Bước 0: Chuẩn Bị Môi Trường

Dự án được thiết kế chạy trên **Google Colab** (Free tier với GPU T4).

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Tạo thư mục làm việc trên Drive
# /content/drive/MyDrive/RAG + Fine_Tuning LLM/
```

### Bước 1: Cài Đặt Thư Viện

```bash
# Fine-tuning
pip install unsloth
pip install --no-deps xformers trl peft accelerate bitsandbytes
pip install datasets

# RAG
pip install langchain==0.1.20 \
            langchain-community==0.0.38 \
            langchain-core==0.1.52 \
            langchain-text-splitters==0.0.1 \
            sentence-transformers \
            faiss-cpu \
            rank_bm25 \
            underthesea

# Evaluation
pip install evaluate rouge_score bert_score sacrebleu

# Demo
pip install gradio
```

### Bước 2: Chuẩn Bị Dữ Liệu

```python
# Upload 3 file luật lên Google Drive:
# - luat_chung_khoan.doc.txt
# - luat_ke_toan.doc.txt
# - luat_thue_tncn.doc.txt

# Chạy cell "Xử lý dữ liệu" trong notebook:
# 1. Làm sạch và đánh giá chất lượng dữ liệu
# 2. Chunking theo Điều luật → knowledge_chunks.json
# 3. Chia train/test splits
```

### Bước 3: Tạo Dataset QA

```python
# Cấu hình DeepSeek API key trong notebook:
API_KEY = "your_deepseek_api_key_here"

# Chạy cell tạo QA (có thể mất vài giờ)
# Output: finance_qa_train.jsonl (≥300 cặp)
#         finance_qa_test.jsonl (50 cặp)
```

> **Lưu ý:** Bước này sử dụng DeepSeek API. Bạn cần đăng ký tài khoản tại [platform.deepseek.com](https://platform.deepseek.com).

### Bước 4: Fine-tuning

```python
# Chạy section "Fine Tuning" trong notebook
# Gồm:
# - Load Qwen2.5-7B-Instruct (4-bit)
# - Cấu hình LoRA adapter
# - Format dataset với prompt template
# - Training 200 steps
# - Lưu adapter → lora_model_finetuned/
```

Thời gian ước tính: ~30–60 phút trên GPU T4.

### Bước 5: Xây Dựng RAG Pipeline

```python
# Chạy section "RAG" trong notebook
# Gồm:
# 1. Load knowledge_chunks.json
# 2. Chunking với RecursiveCharacterTextSplitter
# 3. Embedding với paraphrase-multilingual-mpnet-base-v2
# 4. Build FAISS index → faiss_index/
# 5. Tạo BM25 retriever (với underthesea tokenizer)
# 6. Kết hợp Hybrid Ensemble Retriever
# 7. Load Cross-Encoder Reranker
```

### Bước 6: Chạy Thực Nghiệm

```python
# Chạy section "Benchmark" trong notebook
# Tự động sinh câu trả lời 4 cấu hình A/B/C/D
# Output: Ket_Qua_Thuc_Nghiem_4_Cau_Hinh.csv
```

### Bước 7: Đánh Giá

```python
# Chạy section "Đánh giá" trong notebook
# Tính ROUGE-L, BLEU, BERTScore, Recall@5
# Output: Bang_Diem_Danh_Gia.csv
#         So_Sanh_Theo_Chi_So.png
```

### Bước 8: Load Model Đã Train (Để Dùng Lại)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finance_lora_adapter",  # hoặc đường dẫn HuggingFace Hub
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```

---

## 💻 Yêu Cầu Hệ Thống

| Thành phần | Yêu cầu tối thiểu |
|------------|------------------|
| **GPU** | NVIDIA T4 (16GB VRAM) – Google Colab Free |
| **RAM** | ≥ 12GB |
| **Disk** | ≥ 10GB (model + data + index) |
| **Python** | 3.10+ |
| **CUDA** | 11.8+ |
| **Platform** | Google Colab (khuyến nghị) |

### Thư Viện Chính

| Thư viện | Phiên bản | Chức năng |
|---------|-----------|----------|
| `unsloth` | latest | Fast QLoRA fine-tuning |
| `trl` | — | SFTTrainer |
| `peft` | — | LoRA adapter |
| `bitsandbytes` | — | 4-bit quantization |
| `langchain` | 0.1.20 | RAG orchestration |
| `langchain-community` | 0.0.38 | FAISS, BM25, CrossEncoder |
| `sentence-transformers` | latest | Embedding model |
| `faiss-cpu` | latest | Vector store |
| `rank_bm25` | latest | BM25 retrieval |
| `underthesea` | latest | Vietnamese NLP tokenizer |
| `evaluate` | latest | ROUGE, BLEU, BERTScore |
| `gradio` | latest | Demo UI |
| `datasets` | latest | HuggingFace dataset |
| `torch` | 2.x | Deep learning framework |

---

## 📄 Giấy Phép

Dự án phục vụ mục đích học thuật.

Các văn bản luật sử dụng là tài liệu công khai của Nhà nước Việt Nam.

---

<p align="center">
  <i>Xây dựng với ❤️ để hỗ trợ tra cứu pháp luật tài chính Việt Nam</i>
</p>
