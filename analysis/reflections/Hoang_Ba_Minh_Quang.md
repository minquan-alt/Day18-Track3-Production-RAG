# Individual Reflection — Lab 18

**Tên:** Hoàng Bá Minh Quang  
**Module phụ trách:** M1, M2, M3, M4 và M5 bonus

## 1. Đóng góp kỹ thuật

- Em đến trễ nên xin phép làm Lab một mình, vì vậy tự implement và tự ghép toàn bộ pipeline.
- Module đã implement: chunking đọc PDF/markdown, BM25 + dense fallback, reranker fallback, RAGAS/heuristic eval, enrichment offline.
- Số tests pass: 37/37.

## 2. Kiến thức học được

- Em hiểu rõ hơn vì sao production RAG cần nhiều bước hơn naive RAG: chunking, hybrid search, rerank, eval và failure analysis.
- Điều bất ngờ là PDF scan gần như không extract được text, nên pipeline có thể chạy đúng code nhưng vẫn không có dữ liệu.
- Phần kết nối với bài giảng là hybrid retrieval và đánh giá bằng RAGAS/failure tree.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: Qdrant/RAGAS/model thật không phải lúc nào cũng sẵn, PDF cũng khó đọc.
- Cách giải quyết: viết fallback nhẹ bằng BM25/lexical scoring và thêm notes trích xuất tối thiểu để chạy test_set thật.

## 4. Nếu làm lại

- Em sẽ OCR PDF từ đầu để corpus sát tài liệu gốc hơn.
- Em muốn thử tiếp phần metadata filter và reranker thật bằng BGE reranker.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | * (em làm một mình nên xin phép không tự đánh gái teamwork) | 
| Problem solving | 5 |
