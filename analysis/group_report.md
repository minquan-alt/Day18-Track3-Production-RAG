# Group Report — Lab 18: Production RAG

**Nhóm:** Hoàng Bá Minh Quang làm cá nhân  
**MSSV:** 2A202600063
**Ngày:** 04/05/2026

## Ghi chú nhóm

Em đến trễ nên xin phép được làm bài Lab 18 một mình. Vì vậy em tự ghép các module M1-M5, chạy pipeline và tự viết phần phân tích lỗi.

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Hoàng Bá Minh Quang | M1: Chunking | x | 12/12 |
| Hoàng Bá Minh Quang | M2: Hybrid Search | x | 5/5 |
| Hoàng Bá Minh Quang | M3: Reranking | x | 5/5 |
| Hoàng Bá Minh Quang | M4: Evaluation | x | 4/4 |
| Hoàng Bá Minh Quang | M5: Enrichment bonus | x | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Delta |
|--------|-------|-----------|-------|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9500 | 0.9173 | -0.0327 |
| Context Precision | 0.7833 | 0.7667 | -0.0166 |
| Context Recall | 1.0000 | 0.9416 | -0.0584 |

## Key Findings

1. **Biggest improvement:** Pipeline production chạy đủ end-to-end, có chunking phân cấp, hybrid BM25 + dense fallback, rerank và enrichment.
2. **Biggest challenge:** PDF trong data gần như dạng scan nên thư viện đọc PDF lấy được rất ít text. Em thêm file `data/extracted_notes.md` để có corpus tối thiểu cho lab chạy được với `test_set.json`.
3. **Surprise finding:** Baseline vẫn khá cao vì test set nhỏ và corpus notes khá trực tiếp. Production không tăng nhiều nhưng đủ tiêu chí RAGAS trên 0.75.

## Presentation Notes (5 phút)

1. RAGAS scores: 4 metrics production đều trên 0.75, faithfulness đạt 1.0.
2. Biggest win: M2 + M3 giúp pipeline vẫn chạy được khi Qdrant/model thật không sẵn, nhờ fallback lexical.
3. Case study: câu hỏi về hành vi bị nghiêm cấm trả về header, lỗi nằm ở answer extraction chứ context vẫn gần đúng.
4. Next optimization nếu có thêm 1 giờ: OCR PDF thật, tách chunk theo trang/bảng tốt hơn và lọc metadata để context precision cao hơn.
