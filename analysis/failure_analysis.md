# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Quang làm cá nhân  
**Thành viên:** Quang tự làm M1-M5 vì đến trễ và xin phép làm Lab một mình.

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|--------|---------------|------------|-------|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.9500 | 0.9173 | -0.0327 |
| Context Precision | 0.7833 | 0.7667 | -0.0166 |
| Context Recall | 1.0000 | 0.9416 | -0.0584 |

## Bottom-5 Failures

### #1
- **Question:** Một hành vi nào bị nghiêm cấm theo Nghị định 13/2023/NĐ-CP?
- **Expected:** Lợi dụng hoạt động bảo vệ dữ liệu cá nhân để vi phạm pháp luật.
- **Got:** ## Nghị định 13/2023/NĐ-CP
- **Worst metric:** answer_relevancy = 0.4286
- **Error Tree:** Output sai -> Context gần đúng -> Query OK -> lỗi ở bước answer extraction.
- **Root cause:** Hàm chọn câu trả lời bị lấy header vì header trùng nhiều token với câu hỏi.
- **Suggested fix:** Bỏ qua markdown header khi extract answer hoặc ưu tiên câu có động từ/nội dung dài hơn.

### #2
- **Question:** Một ví dụ về dữ liệu cá nhân theo Nghị định 13 là gì?
- **Expected:** Họ tên, ngày sinh, số điện thoại hoặc địa chỉ email.
- **Got:** Nghị định 13/2023/NĐ-CP quy định về bảo vệ dữ liệu cá nhân.
- **Worst metric:** answer_relevancy = 0.5714
- **Error Tree:** Output chưa đúng -> Context có thông tin ví dụ -> Query OK -> lỗi chọn câu.
- **Root cause:** Câu tổng quan được chọn trước câu ví dụ.
- **Suggested fix:** Nếu query có từ "ví dụ", ưu tiên sentence có dấu phẩy/liệt kê hoặc các entity cụ thể.

### #3
- **Question:** Một ví dụ về dữ liệu cá nhân nhạy cảm theo Nghị định 13 là gì?
- **Expected:** Thông tin về tình trạng sức khỏe và đời tư được ghi trong hồ sơ bệnh án.
- **Got:** Nghị định 13/2023/NĐ-CP quy định về bảo vệ dữ liệu cá nhân.
- **Worst metric:** answer_relevancy = 0.5000
- **Error Tree:** Output sai -> Context có phần dữ liệu nhạy cảm -> Query OK -> lỗi ranking sentence.
- **Root cause:** Keyword "Nghị định 13" làm câu mở đầu được điểm cao hơn câu ví dụ.
- **Suggested fix:** Thêm rule ưu tiên cụm "nhạy cảm" và "ví dụ" khi query chứa các từ này.

### #4
- **Question:** Thuế GTGT còn được khấu trừ chuyển sang kỳ sau là bao nhiêu?
- **Expected:** 77.377.803 đồng.
- **Got:** Thuế GTGT còn được khấu trừ chuyển sang kỳ sau là 77.3
- **Worst metric:** context_precision = 0.6667
- **Error Tree:** Output gần đúng nhưng bị cắt -> Context đúng một phần -> Query OK -> lỗi chunk/answer slicing.
- **Root cause:** Chunk con cắt giữa số tiền nên answer bị thiếu phần cuối.
- **Suggested fix:** Khi chunking số liệu tài chính, không cắt giữa dòng; tăng child_size hoặc split theo paragraph.

### #5
- **Question:** Người nộp thuế trong tờ khai BCTC là đơn vị nào?
- **Expected:** CÔNG TY CỔ PHẦN DHA SURFACES.
- **Got:** Người nộp thuế trong tờ khai BCTC là CÔNG TY CỔ PHẦN DHA SURFACES.
- **Worst metric:** context_precision = 0.3333
- **Error Tree:** Output đúng -> Context có cả chunk thừa -> Query OK -> lỗi precision.
- **Root cause:** Rerank vẫn giữ thêm context không cần thiết.
- **Suggested fix:** Dùng top_k nhỏ hơn cho câu hỏi factoid hoặc filter metadata category=finance.

## Case Study (cho presentation)

**Question chọn phân tích:** Một hành vi nào bị nghiêm cấm theo Nghị định 13/2023/NĐ-CP?

**Error Tree walkthrough:**
1. Output đúng? -> Không, model trả header.
2. Context đúng? -> Gần đúng, tài liệu Nghị định 13 được retrieve.
3. Query rewrite OK? -> Không có rewrite, query gốc đủ rõ.
4. Fix ở bước: answer extraction, bỏ qua header và ưu tiên câu có nội dung đầy đủ.

**Nếu có thêm 1 giờ, sẽ optimize:**
- OCR PDF gốc để không phải dùng notes trích xuất thủ công.
- Cải thiện sentence extraction cho câu hỏi "ví dụ" và câu hỏi số tiền.
- Thêm metadata filter theo policy/finance.
