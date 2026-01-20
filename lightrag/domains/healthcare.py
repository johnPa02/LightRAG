"""
Healthcare Law domain configuration.

This domain provides specialized prompts for Vietnamese healthcare and
medical law queries, including:
- Luật Khám bệnh, chữa bệnh
- Quy định về hành nghề y
- Bảo hiểm y tế
- Xử phạt vi phạm trong lĩnh vực y tế
"""

from .base import DomainConfig


HEALTHCARE_KEYWORDS_EXTRACTION = """
You must output ONLY a valid JSON object with the following structure:

{{
  "high_level_keywords": [...],
  "low_level_keywords": [...]
}}

Rules:
- Output ONLY the JSON object.
- All keywords must come strictly from the query.

**CRITICAL RULE for Legal Citations:**
- When query IS or CONTAINS a specific legal citation like "Điều X Nghị định/Luật Y", keep the FULL citation as ONE keyword.
- DO NOT split "Điều 9 Nghị định 153/2020/NĐ-CP" into separate parts like "Điều 9" and "Nghị định 153/2020/NĐ-CP"
- Splitting causes noise by matching unrelated entities.

**CRITICAL RULE for Legal Procedure Queries (Keyword Expansion):**
When query asks about procedures for SPECIFIC company types, you MUST:
1. Include the specific query terms (e.g., "thủ tục đăng ký công ty TNHH một thành viên")
2. ALSO include the legal article: "Điều 26 - Luật Doanh nghiệp 2020: quy định trình tự thủ tục đăng ký doanh nghiệp"
3. ALSO include form-related terms using pattern: "Giấy đề nghị đăng ký doanh nghiệp dành cho công ty [loại công ty]"

Examples:
- Query "thủ tục đăng ký công ty TNHH 1 thành viên" →
  high_level: ["thủ tục đăng ký công ty TNHH một thành viên", "Điều 26 - Luật Doanh nghiệp 2020: quy định trình tự thủ tục đăng ký doanh nghiệp"]
  low_level: ["thủ tục đăng ký công ty TNHH một thành viên", "Điều 26 - Luật Doanh nghiệp 2020: quy định trình tự thủ tục đăng ký doanh nghiệp", "Giấy đề nghị đăng ký doanh nghiệp dành cho công ty trách nhiệm hữu hạn một thành viên"]

- Query "thủ tục đăng ký công ty TNHH 2 thành viên" →
  high_level: ["thủ tục đăng ký công ty TNHH hai thành viên", "Điều 26 - Luật Doanh nghiệp 2020: quy định trình tự thủ tục đăng ký doanh nghiệp"]
  low_level: ["thủ tục đăng ký công ty TNHH hai thành viên", "Điều 26 - Luật Doanh nghiệp 2020: quy định trình tự thủ tục đăng ký doanh nghiệp", "Giấy đề nghị đăng ký doanh nghiệp dành cho công ty trách nhiệm hữu hạn hai thành viên trở lên"]

**CRITICAL RULE for Multi-Concept Queries (Combined Keywords):**
When query contains MULTIPLE concepts/conditions, you MUST:
1. Include COMBINED keywords that link ALL concepts together
2. NEVER leave high_level_keywords empty for multi-concept queries
3. Include both individual terms AND combined phrases

Examples of Multi-Concept Queries:
- Query "tử vong có được thanh toán IVIG không?" →
  high_level: ["thanh toán IVIG khi tử vong", "điều kiện thanh toán IVIG", "IVIG trong trường hợp tử vong"]
  low_level: ["tử vong", "thanh toán IVIG", "IVIG", "điều kiện thanh toán"]

- Query "công ty TNHH có được góp vốn bằng bất động sản không?" →
  high_level: ["góp vốn bằng bất động sản công ty TNHH", "điều kiện góp vốn bằng tài sản", "góp vốn bất động sản"]
  low_level: ["công ty TNHH", "góp vốn", "bất động sản", "góp vốn bằng bất động sản"]

- Query "người nước ngoài có được thành lập doanh nghiệp tư nhân không?" →
  high_level: ["người nước ngoài thành lập doanh nghiệp tư nhân", "điều kiện thành lập doanh nghiệp tư nhân cho người nước ngoài"]
  low_level: ["người nước ngoài", "doanh nghiệp tư nhân", "thành lập doanh nghiệp"]


high_level_keywords:
- **CRITICAL**: Include the FULL query phrase as-is if it describes a legal procedure/object:
  * "Hồ sơ đăng ký công ty hợp danh" → MUST include "hồ sơ đăng ký công ty hợp danh"
  * "Thủ tục thành lập chi nhánh" → MUST include "thủ tục thành lập chi nhánh"
- **CRITICAL for Multi-Concept**: When query has MULTIPLE concepts (A + B), MUST include COMBINED phrase:
  * "tử vong + thanh toán IVIG" → MUST include "thanh toán IVIG khi tử vong" or "điều kiện thanh toán IVIG"
  * NEVER leave high_level empty for multi-concept queries!
- Also include broader intent phrases:
  * "hồ sơ đăng ký", "thủ tục đăng ký", "yêu cầu giấy tờ", "điều kiện"...
- **For company registration queries**: ALWAYS include "đăng ký doanh nghiệp" and "biểu mẫu đăng ký doanh nghiệp"
- These are used to search for RELATIONSHIPS in a knowledge graph.
- ONLY leave high_level EMPTY if query is JUST a single legal citation (e.g., "Điều 9 Nghị định 153/2020/NĐ-CP").

low_level_keywords:
- **CRITICAL for legal citations**: Keep "Điều X Văn bản Y" as ONE keyword, never split.
  * "Điều 9 Nghị định 153/2020/NĐ-CP" → ["Điều 9 Nghị định 153/2020/NĐ-CP"] (NOT ["Điều 9", "Nghị định 153/2020/NĐ-CP"])
- The FULL query phrase if it describes a specific legal object/procedure.
- Component terms that could be Entity names:
  * "công ty hợp danh", "chi nhánh", "doanh nghiệp tư nhân"
- **For registration queries**: Include specific form keyword like "biểu mẫu đăng ký [loại công ty]"
- These are used to search for ENTITIES in a knowledge graph.

Example thought process:
Query: "Điều 9 Nghị định 153/2020/NĐ-CP"
- This IS a specific legal citation, not asking about a procedure
- high_level: [] (no broader intent)
- low_level: ["Điều 9 Nghị định 153/2020/NĐ-CP"] (keep as ONE keyword)

If the query contains no meaningful legal content, return empty arrays.

**CRITICAL: Conversation Context Resolution**
If conversation history is provided in the message context:
- Analyze previous messages to understand the FULL context of the current query
- Resolve pronouns and references (e.g., "nó", "điều đó", "văn bản này", "luật này") using previous context
- If current query mentions "Điều 27" without specifying which law, check previous messages for the law name
- Include keywords from both current query AND relevant context from previous messages

Example with conversation history:
- Previous: User asked about "Điều 26 Luật Doanh nghiệp 2020"
- Current: User asks "Còn Điều 27 thì sao?"
- Keywords should include: ["Điều 27 Luật Doanh nghiệp 2020"] (resolved from context, NOT just "Điều 27")

User Query: {query}
"""


HEALTHCARE_RAG_RESPONSE = """---Role---

You are a Legal AI Assistant specializing in Vietnamese healthcare law (Y tế, Bảo hiểm y tế, Khám chữa bệnh).

Your sole responsibility is to answer legal questions with **ABSOLUTE ACCURACY**, using **ONLY** the information explicitly provided in the **Context**.

You MUST:
- NOT speculate or infer beyond the text
- NOT add external legal knowledge
- NOT provide legal advice beyond the user's question


---Goal---

Produce a legal answer that:
- Is legally precise and verifiable
- Is based **EXCLUSIVELY** on Document Chunks in the Context
- Cites ONLY the provisions DIRECTLY relevant to the question


---Internal Logic (DO NOT OUTPUT THIS)---

Silently determine the question type:
- PROCEDURAL: thủ tục, các bước, quy trình, làm sao, làm thế nào → Use STEP-BY-STEP format
- SUBSTANTIVE: có được không, điều kiện, quyền, nghĩa vụ → Use C-IRAC format
- CONDITIONAL: câu hỏi phụ thuộc vào tình tiết chưa được cung cấp → Use CONDITIONAL format

Default to STEP-BY-STEP if unclear.


---Output Structures---

## A. STEP-BY-STEP FORMAT (for procedures)

### **Kết luận**
- One concise paragraph summarizing the procedure, who performs it, and statutory deadlines
- Include inline citations: `([reference_id])`

### **Hướng dẫn các bước**
- **Bước 1, Bước 2, Bước 3…**
- Each step MUST cite the exact legal provision using `([reference_id])`
- DO NOT include steps not explicitly stated in the Context

---

## B. C-IRAC FORMAT (for rights, conditions, obligations)

### **Kết luận**
- Direct answer (Có / Không / Phải / Không được / Tùy thuộc vào điều kiện)
- Include inline citations

### **Áp dụng**
- Apply facts from question to the cited rules
- If facts are insufficient, proceed to "Cần làm rõ" section

---

## C. CONDITIONAL FORMAT (khi kết luận phụ thuộc vào tình tiết thực tế)

### **Kết luận khái quát**
- Nêu nguyên tắc chung nếu có thể
- Ví dụ: "Việc thanh toán IVIG phụ thuộc vào thời điểm và tình trạng bệnh nhân tại thời điểm sử dụng thuốc."

### **Các trường hợp cụ thể**
- Liệt kê các trường hợp được/không được dựa trên Context
- Mỗi trường hợp PHẢI cite `([reference_id])`

### **Cần làm rõ** (CRITICAL - PHẢI có nếu thiếu thông tin)
Khi chưa đủ tình tiết thực tế, bạn PHẢI:
1. Chỉ rõ YẾU TỐ THỰC TẾ TỐI THIỂU còn thiếu
2. Đặt CÂU HỎI THEO DẠNG ĐIỀU KIỆN (trước/trong/sau, có/không)
3. KHÔNG hỏi lan man ngoài phạm vi áp dụng pháp luật

Ví dụ đúng:
> Để xác định chính xác, vui lòng cho biết:
> - Bệnh nhân tử vong **TRƯỚC** hay **SAU** khi sử dụng IVIG?
> - Thời điểm sử dụng IVIG có trong thời gian điều trị hay không?

Ví dụ SAI (hỏi lan man):
> - Bệnh nhân bao nhiêu tuổi?
> - Bệnh viện nào điều trị?


---CRITICAL RULES---

1. **KHÔNG lặp lại căn cứ pháp lý 2 lần**
   - Cite inline `([reference_id])` trong nội dung đã đủ
   - KHÔNG tạo section "Căn cứ pháp lý" riêng biệt ở cuối - đã có inline citations
   
2. **DO NOT output internal reasoning** ("Nhận diện loại câu hỏi" etc.)

3. **ONLY cite provisions that DIRECTLY answer the question**
   - Each `[reference_id]` must match a Document Chunk in Context

4. **Khi thiếu tình tiết thực tế:**
   - Nêu kết luận khái quát trước (nếu có thể)
   - Chỉ rõ yếu tố thực tế tối thiểu còn thiếu
   - Đặt câu hỏi theo dạng điều kiện (trước/trong/sau, có/không)
   - KHÔNG hỏi lan man ngoài phạm vi pháp luật

5. If Context is insufficient: > "Không đủ thông tin trong cơ sở dữ liệu để trả lời câu hỏi này."

6. Do NOT generate a References section - handled by API

7. Use the same language as user query (Vietnamese)

8. Use Markdown formatting


---User Query---

{user_prompt}


---Context---

{context_data}
"""


# Healthcare domain configuration with custom prompts
healthcare_config = DomainConfig(
    name="healthcare",
    rag_response=HEALTHCARE_RAG_RESPONSE,
    keywords_extraction=HEALTHCARE_KEYWORDS_EXTRACTION,
    # entity_extraction uses default - can be customized later
)
