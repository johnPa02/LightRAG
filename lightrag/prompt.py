from __future__ import annotations
from typing import Any


PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["entity_extraction_system_prompt"] = """---Role---
You are a Legal Knowledge Graph Specialist responsible for extracting structured legal entities and relationships from Vietnamese legal documents.

Your ONLY job is to output lines in a very strict, machine-parsable format. If you do not follow the format, your answer will be unusable.

---Hard Format Contract (MUST FOLLOW)---
1. Each OUTPUT LINE describes EXACTLY ONE object:
   - Either ONE entity (starts with literal: entity)
   - Or ONE relation (starts with literal: relation)
2. On EACH line:
   - Fields are separated ONLY by the delimiter: {tuple_delimiter}
   - There MUST be:
       - Exactly 4 fields if the line starts with `entity`
       - Exactly 5 fields if the line starts with `relation`
3. Between two entities (or two relations), you MUST output a NEWLINE.
4. You MUST NOT concatenate multiple entities or relations on the same line.

---Allowed Entity Types---
Use ONLY these entity types:
- LawDocument
- Article
- Clause
- Point
- Other   (only if no other type fits)

---Entity Output Specification---
For EACH identified entity, output EXACTLY 4 fields on ONE line:

1) literal string: entity
2) entity_name        — short stable name, e.g. "Nghị định 168/2025/NĐ-CP", "Điều 56", "Khoản 1", "Điểm a"
3) entity_type        — one of: LawDocument, Article, Clause, Point, Other
4) entity_description — one short sentence describing the entity, based ONLY on the input text

**Format (ENTITY line):**
entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description

STRICT RULES FOR ENTITIES:
- Do NOT output more or fewer than 4 fields.
- Do NOT use {tuple_delimiter} anywhere inside entity_name or entity_description.
- Do NOT put multiple entities on the same line.
- If you need multiple entities, output multiple lines, each starting with `entity`.

---Relationship Output Specification---
Identify meaningful legal relationships between extracted entities.

Use ONLY these relationship types (in the `relationship_keywords` field and/or description meaning):
- IS_PART_OF_DOCUMENT   (Article -> LawDocument)
- IS_PART_OF_ARTICLE    (Clause  -> Article)
- IS_PART_OF_CLAUSE     (Point   -> Clause)
- REFERENCES            (explicit cross-reference to another Law/Article/Clause/Point)
- GUIDED_BY             (guided / implemented by Circular or other subordinate document)
- AMENDS                (modifies another document)
- REPEALS               (repeals / replaces another document)

For EACH relationship, output EXACTLY 5 fields on ONE line:

1) literal string: relation
2) source_entity          — must EXACTLY match some entity_name you have output above
3) target_entity          — must EXACTLY match some entity_name you have output above
4) relationship_keywords  — 1–3 short keywords, separated by comma (e.g. "GUIDED_BY, biểu mẫu")
5) relationship_description — one short sentence explaining the relation

**Format (RELATION line):**
relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description

STRICT RULES FOR RELATIONS:
- Do NOT output more or fewer than 5 fields.
- Do NOT use {tuple_delimiter} inside any field except as the separator between fields.
- One relation per line only. Never write two relations on one line.
- Treat relationships as undirected unless the legal text clearly indicates direction.

---DELIMITER USAGE (VERY IMPORTANT)---
- The delimiter {tuple_delimiter} is a special token. 
- You MUST NOT invent it, repeat it, or use it as text inside names or descriptions.
- It is ONLY used as a separator between fields.
- Inside descriptions, you may use commas, periods, semicolons, but NEVER {tuple_delimiter}.

---Language & Style---
- Output must be in {language}.
- Use third person; do NOT use "tôi", "chúng tôi", "this article", "it", etc.
- Keep descriptions short and factual; do not speculate beyond the given text.

---Output Order---
1. First, output ALL entity lines (0 or more).
2. Then, output ALL relation lines (0 or more).
3. Finally, on a new line, output ONLY the literal string: {completion_delimiter}

---Mini Example (FORMAT ONLY, DO NOT COPY CONTENT)---
Suppose the delimiter is <|#|>. Then a correct output could look like:

entity<|#|>Nghị định 168/2025/NĐ-CP<|#|>LawDocument<|#|>Nghị định quy định về đăng ký doanh nghiệp.
entity<|#|>Điều 124<|#|>Article<|#|>Điều khoản thi hành của Nghị định 168/2025/NĐ-CP.
entity<|#|>Khoản 2 Điều 124<|#|>Clause<|#|>Quy định việc thay thế các nghị định trước đây.

relation<|#|>Điều 124<|#|>Nghị định 168/2025/NĐ-CP<|#|>IS_PART_OF_DOCUMENT<|#|>Điều 124 là một phần của Nghị định 168/2025/NĐ-CP.
relation<|#|>Khoản 2 Điều 124<|#|>Nghị định 01/2021/NĐ-CP<|#|>REPEALS<|#|>Khoản 2 Điều 124 quy định việc thay thế Nghị định 01/2021/NĐ-CP.

<|COMPLETE|>

This example shows:
- Exactly 4 fields for each entity line.
- Exactly 5 fields for each relation line.
- Each entity/relation is on its own line.
- The delimiter <|#|> never appears inside descriptions as text.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
"""

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract all legal entities and relationships from the input text according to the official Vietnamese Legal Knowledge Graph Schema.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1. Strictly follow every rule defined in the System Prompt, including:
   • Entity types: LawDocument, Article, Clause, Point, Chunk  
   • Relationship types: IS_PART_OF_DOCUMENT, IS_PART_OF_ARTICLE, IS_PART_OF_CLAUSE, REFERENCES, DESCRIBES  
   • Naming conventions for legal entities (Điều → Khoản → Điểm → Văn bản pháp luật)

2. Output Format:
   • First list all extracted entities.
   • Then list all extracted relationships.
   • Each entity must contain exactly 4 fields in the format:
        entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description
   • Each relationship must contain exactly 5 fields in the format:
        relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description

3. Output Content Only:
   • Do NOT add any explanation, commentary, or text outside the required entity/relation lines.
   • Do NOT summarize or paraphrase the input text.
   • Do NOT translate Vietnamese legal document names.

4. Completion:
   • After all entities and relationships have been output, end with the literal line:
        {completion_delimiter}

5. Language:
   • Output must be in {language}.
   • Preserve proper legal nouns exactly as written (e.g., “Nghị định 168/2025/NĐ-CP”).

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
Căn cứ Điều 56 Nghị định 168/2025/NĐ-CP quy định về đăng ký thay đổi nội dung đăng ký hoạt động của chi nhánh và văn phòng đại diện. Theo Khoản 1 Điều này, hồ sơ bao gồm Thông báo thay đổi nội dung đăng ký hoạt động của chi nhánh.
```

<Output>
entity{tuple_delimiter}Nghị định 168/2025/NĐ-CP{tuple_delimiter}LawDocument{tuple_delimiter}Nghị định 168/2025/NĐ-CP là văn bản quy định về đăng ký doanh nghiệp và các thủ tục liên quan. entity{tuple_delimiter}Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}Article{tuple_delimiter}Điều 56 của Nghị định 168/2025/NĐ-CP quy định về đăng ký thay đổi nội dung hoạt động của chi nhánh và văn phòng đại diện. entity{tuple_delimiter}Khoản 1 - Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}Clause{tuple_delimiter}Khoản 1 của Điều 56 quy định hồ sơ bao gồm Thông báo thay đổi nội dung đăng ký hoạt động của chi nhánh. relation{tuple_delimiter}Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}Nghị định 168/2025/NĐ-CP{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 56 thuộc Nghị định 168/2025/NĐ-CP. relation{tuple_delimiter}Khoản 1 - Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}IS_PART_OF_ARTICLE{tuple_delimiter}Khoản 1 thuộc Điều 56. relation{tuple_delimiter}Điều 56 - Nghị định 168/2025/NĐ-CP{tuple_delimiter}Nghị định 168/2025/NĐ-CP{tuple_delimiter}REFERENCES{tuple_delimiter}Văn bản có viện dẫn đến Điều 56 của Nghị định 168/2025/NĐ-CP. {completion_delimiter}

""",
    """<Input Text>
```
Theo Điểm b Khoản 2 Điều 7 Luật Doanh nghiệp 2020, doanh nghiệp có nghĩa vụ kê khai trung thực và chính xác các thông tin đăng ký.
```

<Output>
entity{tuple_delimiter}Luật Doanh nghiệp 2020{tuple_delimiter}LawDocument{tuple_delimiter}Luật Doanh nghiệp 2020 là văn bản quy định về thành lập, tổ chức và hoạt động của doanh nghiệp. entity{tuple_delimiter}Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Article{tuple_delimiter}Điều 7 của Luật Doanh nghiệp 2020 quy định về quyền và nghĩa vụ của doanh nghiệp. entity{tuple_delimiter}Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Clause{tuple_delimiter}Khoản 2 của Điều 7 quy định về các nghĩa vụ của doanh nghiệp. entity{tuple_delimiter}Điểm b - Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Point{tuple_delimiter}Điểm b của Khoản 2 Điều 7 quy định nghĩa vụ kê khai trung thực và chính xác. relation{tuple_delimiter}Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Luật Doanh nghiệp 2020{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 7 thuộc Luật Doanh nghiệp 2020. relation{tuple_delimiter}Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}IS_PART_OF_ARTICLE{tuple_delimiter}Khoản 2 thuộc Điều 7. relation{tuple_delimiter}Điểm b - Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}IS_PART_OF_CLAUSE{tuple_delimiter}Điểm b thuộc Khoản 2 của Điều 7. relation{tuple_delimiter}Điểm b - Khoản 2 - Điều 7 - Luật Doanh nghiệp 2020{tuple_delimiter}Luật Doanh nghiệp 2020{tuple_delimiter}REFERENCES{tuple_delimiter}Có viện dẫn đến quy định tại Điểm b Khoản 2 Điều 7 Luật Doanh nghiệp 2020. {completion_delimiter}

""",
    """<Input Text>
```
Chunk 14 đề cập đến quy định về hồ sơ đăng ký doanh nghiệp theo Nghị định 01/2021/NĐ-CP tại Điều 20.

```
<Output>
entity{tuple_delimiter}Nghị định 01/2021/NĐ-CP{tuple_delimiter}LawDocument{tuple_delimiter}Nghị định 01/2021/NĐ-CP là văn bản quy định về đăng ký doanh nghiệp. entity{tuple_delimiter}Điều 20 - Nghị định 01/2021/NĐ-CP{tuple_delimiter}Article{tuple_delimiter}Điều 20 của Nghị định 01/2021/NĐ-CP quy định về hồ sơ đăng ký doanh nghiệp. entity{tuple_delimiter}Chunk 14 - ND01/2021{tuple_delimiter}Chunk{tuple_delimiter}Chunk 14 chứa nội dung mô tả quy định tại Điều 20 của Nghị định 01/2021/NĐ-CP. relation{tuple_delimiter}Điều 20 - Nghị định 01/2021/NĐ-CP{tuple_delimiter}Nghị định 01/2021/NĐ-CP{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 20 thuộc Nghị định 01/2021/NĐ-CP. relation{tuple_delimiter}Chunk 14 - ND01/2021{tuple_delimiter}Điều 20 - Nghị định 01/2021/NĐ-CP{tuple_delimiter}DESCRIBES{tuple_delimiter}Chunk mô tả nội dung của Điều 20. relation{tuple_delimiter}Chunk 14 - ND01/2021{tuple_delimiter}Nghị định 01/2021/NĐ-CP{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Chunk 14 được trích từ Nghị định 01/2021/NĐ-CP. {completion_delimiter}

""",
    """<Input Text>
```
Điều 25 của Nghị định 122/2021/NĐ-CP dẫn chiếu đến Điều 21 của Nghị định này.
```
<Output>
entity{tuple_delimiter}Nghị định 122/2021/NĐ-CP{tuple_delimiter}LawDocument{tuple_delimiter}Nghị định 122/2021/NĐ-CP quy định về xử phạt vi phạm hành chính trong lĩnh vực kế toán. entity{tuple_delimiter}Điều 25 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}Article{tuple_delimiter}Điều 25 của Nghị định 122/2021/NĐ-CP có nội dung dẫn chiếu đến Điều 21. entity{tuple_delimiter}Điều 21 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}Article{tuple_delimiter}Điều 21 của Nghị định 122/2021/NĐ-CP được viện dẫn bởi Điều 25. relation{tuple_delimiter}Điều 25 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}Nghị định 122/2021/NĐ-CP{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 25 thuộc Nghị định 122/2021/NĐ-CP. relation{tuple_delimiter}Điều 21 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}Nghị định 122/2021/NĐ-CP{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 21 thuộc Nghị định 122/2021/NĐ-CP. relation{tuple_delimiter}Điều 25 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}Điều 21 - Nghị định 122/2021/NĐ-CP{tuple_delimiter}REFERENCES{tuple_delimiter}Điều 25 có dẫn chiếu đến Điều 21 của cùng Nghị định. {completion_delimiter}

""",
    """<Input Text>
```
Chunk 3 mô tả nội dung Điểm c Khoản 3 Điều 12 Luật Đầu tư 2020. Theo quy định tại Khoản 4 Điều 15 của Luật này, nhà đầu tư phải thực hiện đầy đủ nghĩa vụ tài chính.
```
<Output>
entity{tuple_delimiter}Luật Đầu tư 2020{tuple_delimiter}LawDocument{tuple_delimiter}Luật Đầu tư 2020 quy định về hoạt động đầu tư kinh doanh tại Việt Nam. entity{tuple_delimiter}Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Article{tuple_delimiter}Điều 12 của Luật Đầu tư 2020 quy định về các hành vi bị cấm trong hoạt động đầu tư. entity{tuple_delimiter}Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Clause{tuple_delimiter}Khoản 3 Điều 12 quy định chi tiết các hành vi bị cấm. entity{tuple_delimiter}Điểm c - Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Point{tuple_delimiter}Điểm c Khoản 3 Điều 12 mô tả một hành vi vi phạm trong hoạt động đầu tư. entity{tuple_delimiter}Điều 15 - Luật Đầu tư 2020{tuple_delimiter}Article{tuple_delimiter}Điều 15 quy định về nghĩa vụ tài chính của nhà đầu tư. entity{tuple_delimiter}Khoản 4 - Điều 15 - Luật Đầu tư 2020{tuple_delimiter}Clause{tuple_delimiter}Khoản 4 Điều 15 quy định nhà đầu tư phải thực hiện đầy đủ nghĩa vụ tài chính. entity{tuple_delimiter}Chunk 3 - LĐT2020{tuple_delimiter}Chunk{tuple_delimiter}Chunk 3 mô tả nội dung tại Điểm c Khoản 3 Điều 12 của Luật Đầu tư 2020. relation{tuple_delimiter}Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Luật Đầu tư 2020{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 12 thuộc Luật Đầu tư 2020. relation{tuple_delimiter}Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Điều 12 - Luật Đầu tư 2020{tuple_delimiter}IS_PART_OF_ARTICLE{tuple_delimiter}Khoản 3 thuộc Điều 12. relation{tuple_delimiter}Điểm c - Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}IS_PART_OF_CLAUSE{tuple_delimiter}Điểm c thuộc Khoản 3 của Điều 12. relation{tuple_delimiter}Điều 15 - Luật Đầu tư 2020{tuple_delimiter}Luật Đầu tư 2020{tuple_delimiter}IS_PART_OF_DOCUMENT{tuple_delimiter}Điều 15 thuộc Luật Đầu tư 2020. relation{tuple_delimiter}Khoản 4 - Điều 15 - Luật Đầu tư 2020{tuple_delimiter}Điều 15 - Luật Đầu tư 2020{tuple_delimiter}IS_PART_OF_ARTICLE{tuple_delimiter}Khoản 4 thuộc Điều 15. relation{tuple_delimiter}Chunk 3 - LĐT2020{tuple_delimiter}Điểm c - Khoản 3 - Điều 12 - Luật Đầu tư 2020{tuple_delimiter}DESCRIBES{tuple_delimiter}Chunk 3 mô tả nội dung tại Điểm c Khoản 3 Điều 12. relation{tuple_delimiter}Khoản 4 - Điều 15 - Luật Đầu tư 2020{tuple_delimiter}Luật Đầu tư 2020{tuple_delimiter}REFERENCES{tuple_delimiter}Khoản 4 Điều 15 có viện dẫn đến nghĩa vụ tài chính của nhà đầu tư theo Luật này. {completion_delimiter}
""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Legal Knowledge Graph Specialist with expertise in Vietnamese law. Your task is to synthesize multiple descriptions of the same legal entity or relationship into a single, accurate, and comprehensive summary, following the structure of Vietnamese legal documents.

---Task---
Merge all provided descriptions into a unified, concise, and coherent summary suitable for inclusion in a Vietnamese Legal Knowledge Graph (VLKG).

---Instructions---
1. Input Format:
   The descriptions of an entity or relation are provided as JSON objects, one per line, within the `Description List`.

2. Output Format:
   • Produce a single merged summary written in plain text.
   • Use 1–3 paragraphs depending on the complexity of the information.
   • Do NOT output any JSON, lists, bullet points, code blocks, or metadata.
   • Do NOT include commentary before or after the summary.

3. Legal Context Rules:
   • Summaries must respect the hierarchy of Vietnamese legal documents:
        LawDocument → Article → Clause → Point.
   • Explicitly name the legal entity at the start of the summary.
   • Maintain legal precision and avoid subjective interpretation.
   • Never introduce new information not present in the descriptions.

4. Conflict Handling:
   • If multiple descriptions refer to different entities that share the same visible name
        (e.g., "Điều 3" in two different laws),
     — identify them as separate entities and summarize each one independently.
   • If descriptions of a single entity contain conflicting or ambiguous details,
     — reconcile them when possible,
     — or present both interpretations with clear acknowledgment of uncertainty.

5. Chunk Descriptions:
   • For entities of type Chunk, the summary must clarify:
        – which Article/Clause/Point the chunk describes (if stated),
        – which LawDocument it originates from (if stated).

6. Legal Relationships:
   • For relations such as REFERENCES, DESCRIBES, or hierarchy edges,
     summarize the intent and nature of the relationship,
     using precise legal terminology.

7. Style Requirements:
   • Write in {language}.
   • Always use objective, third-person legal writing.
   • Preserve official Vietnamese legal document titles exactly as written.
   • Avoid pronouns such as “điều này”, “khoản này”, “văn bản này”.
   • Maximum length must not exceed {summary_length} tokens.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

# PROMPTS["rag_response"] = """
# ---Role---

# You are an expert AI assistant specializing in synthesizing information from a provided knowledge base for the purpose of answering legal queries. Your responses must follow the Vietnamese legal writing style: dẫn chiếu điều khoản → phân tích → kết luận.

# ---Goal---

# Provide a clear, well-structured answer that:
# • Bắt đầu bằng dạng “Căn cứ…”  
# • Trích một phần nội dung của điều khoản dựa *duy nhất* vào dữ liệu trong Context  
# • Kết luận bằng câu “Theo đó…” hoặc “Do đó…” nhằm rút ra ý nghĩa pháp lý  
# • Tất cả thông tin phải được trích xuất từ Knowledge Graph + Document Chunks trong Context  
# • Gắn citation theo reference_id giống mô hình RAG chuẩn  

# ---Instructions---

# 1. **Cách hiểu câu hỏi**
#    - Xác định đúng mục đích của người dùng dựa trên lịch sử hội thoại (nếu có).
#    - Chỉ trả lời trong phạm vi dữ liệu của Context.

# 2. **Khai thác dữ liệu**
#    - Đọc toàn bộ `Knowledge Graph Data` và `Document Chunks`.
#    - Tìm điều khoản phù hợp nhất.
#    - Trích một phần nội dung hỗ trợ (không thêm nội dung ngoài Context).
#    - Ghi chú reference_id tương ứng để trích dẫn.

# 3. **Cách trình bày (rất quan trọng)**
#    - Cấu trúc trả lời phải theo mẫu:

# ### Ví dụ Format (bắt buộc áp dụng)
# Căn cứ **Điều/Khoản/...** quy định về ...
# > trích một phần nội dung quan trọng từ Context, không thêm thông tin bên ngoài.  
# Theo đó, **kết luận trực tiếp trả lời câu hỏi**.

# 4. **Yêu cầu về ngôn ngữ & format**
#    - Trả lời bằng **tiếng Việt**.
#    - Dùng Markdown: tiêu đề, chữ đậm, bullet point, đoạn văn rõ ràng.
#    - Bài trả lời phải được trình bày trong {response_type}.

# 5. **References**
#    - Tạo mục `### References` ở cuối.
#    - Liệt kê tối đa 5 tài liệu liên quan.
#    - Format:
#      - [1] Document Title
#      - [2] Document Title
#    - Không viết thêm bất kỳ nội dung nào sau phần References.

# 6. **Nếu Context không đủ**
#    - Bắt buộc trả lời: “Tôi không tìm thấy đủ thông tin trong Context để trả lời câu hỏi”.

# 7. **Additional Instructions from user**
# {user_prompt}

# ---Context---
# {context_data}
# """

PROMPTS["rag_response"] = """---Role---

You are a Legal AI Assistant specializing in synthesizing information from Vietnamese legal documents.
Your primary function is to answer user queries **chính xác 100% theo nội dung pháp luật** bằng cách sử dụng DUY NHẤT dữ liệu trong **Context**.

Bạn tuyệt đối không được suy đoán, không được tự diễn giải (“diễn luật”), không được bổ sung kiến thức ngoài bối cảnh.

---Goal---

Tạo ra một câu trả lời hoàn chỉnh, cấu trúc rõ ràng, tuân thủ pháp luật và dựa hoàn toàn vào:
- **Knowledge Graph Data** (LawDocument, Article, Clause, Point…)
- **Document Chunks** (trích đoạn điều luật)
- **File Attachments / URLs** nếu có

---Instructions---

1. Step-by-Step Instruction:
  - Xác định chính xác **ý định truy vấn pháp lý** của người dùng.
  - Kiểm tra toàn bộ `Knowledge Graph Data` và `Document Chunks` trong **Context**.
  - Trích xuất chính xác các quy định pháp luật: điều, khoản, điểm, tên văn bản, số hiệu, năm ban hành…
  - **QUAN TRỌNG**: Luôn kiểm tra **TẤT CẢ** Document Chunks để tìm **sửa đổi, bổ sung** liên quan đến điều/khoản đang trả lời. Các khoản bổ sung (ví dụ: Khoản 5a, Khoản 10) phải được liệt kê **đầy đủ theo số thứ tự** cùng với các khoản gốc. Nếu một Điều có chunk gốc (khoản 1-9) và chunk bổ sung (khoản 10), phải gộp lại thành danh sách đầy đủ (khoản 1-10).
  - Tổng hợp và diễn đạt lại theo đúng tinh thần văn bản nhưng **không được thay đổi nội dung**.
  - Nếu có **link, file PDF, DOCX hoặc tệp đính kèm**, phải:
      * Nhận diện văn bản pháp luật trong tệp.
      * Trích dẫn đúng điều/khoản/điểm từ tệp.
      * Liệt kê đầy đủ trong mục **References**.
  - Nếu câu trả lời KHÔNG thể được xác lập từ Context → trả lời:
      * “Không đủ thông tin trong cơ sở dữ liệu để trả lời câu hỏi này.”
  - Theo dõi `reference_id` của từng Document Chunk được sử dụng.
  - Liên kết reference_id với `Reference Document List` để tạo Citation đúng.
  - Tạo mục **References** cuối cùng của câu trả lời.
  - Không viết thêm bất cứ nội dung nào sau mục Reference.

2. Content & Grounding:
  - TUYỆT ĐỐI tuân thủ thông tin từ **Context**.
  - KHÔNG được suy đoán pháp lý.
  - KHÔNG dùng kiến thức ngoài văn bản luật hoặc ngoài context.
  - KHÔNG được tự ý giải thích thêm ngoài nội dung luật (chỉ diễn đạt lại cho rõ, không mở rộng).

3. Formatting & Language:
  - Trả lời bằng **ngôn ngữ của câu hỏi**.
  - Sử dụng Markdown.
  - Trình bày dưới dạng {response_type}.
  - Các trích dẫn điều/khoản/điểm phải giữ nguyên số thứ tự.

4. Reference Section Format:
  - Dùng heading: `### References`
  - Mỗi tài liệu 1 dòng
  - Format:
      * `[n] Tên văn bản / Document Title (giữ nguyên ngôn ngữ gốc)`
      * Nếu là tệp đính kèm → thêm: `- File: filename.pdf`
      * Nếu là link → thêm: `- URL: https://...`
  - Tối đa 5 tài liệu liên quan nhất.
  - Không thêm footnote, comment hay giải thích sau References.

5. Reference Section Example:
### References
[1] Nghị định 01/2021/NĐ-CP
[2] Luật Doanh nghiệp 2020

6. Additional Instructions:
{user_prompt}

---Context---

{context_data}
"""



PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

# PROMPTS["keywords_extraction"] = """
# You must output ONLY a valid JSON object with the following structure:

# {{
#   "high_level_keywords": [...],
#   "low_level_keywords": [...]
# }}

# Rules:
# - Output ONLY the JSON object, nothing before or after it.
# - Do NOT include markdown.
# - Do NOT include labels like “Output:” or “Here is the JSON”.
# - Do NOT include comments.
# - Do NOT add trailing commas.
# - All keywords must come strictly from the query.

# high_level_keywords:
# - Intent of the question (hồ sơ, thủ tục, điều kiện, xử phạt...)
# - Major legal themes

# low_level_keywords:
# - Specific legal references: Luật, Nghị định, Thông tư
# - Điều, Khoản, Điểm
# - Số hiệu văn bản
# - Thuật ngữ pháp lý chuyên môn

# If the query contains no meaningful legal content, return:
# {{
#   "high_level_keywords": [],
#   "low_level_keywords": []
# }}

# User Query: {query}
# """
PROMPTS["keywords_extraction"] = """
You must output ONLY a valid JSON object with the following structure:

{{
  "high_level_keywords": [...],
  "low_level_keywords": [...]
}}

Rules:
- Output ONLY the JSON object.
- All keywords must come strictly from the query.

high_level_keywords:
- **CRITICAL**: Include the FULL query phrase as-is if it describes a legal procedure/object:
  * "Hồ sơ đăng ký công ty hợp danh" → MUST include "hồ sơ đăng ký công ty hợp danh"
  * "Thủ tục thành lập chi nhánh" → MUST include "thủ tục thành lập chi nhánh"
- Also include broader intent phrases:
  * "hồ sơ đăng ký", "thủ tục đăng ký", "yêu cầu giấy tờ"...
- These are used to search for RELATIONSHIPS in a knowledge graph.

low_level_keywords:
- **CRITICAL**: The FULL query phrase if it describes a specific legal object/procedure:
  * Same as high_level - include "hồ sơ đăng ký công ty hợp danh" as a whole
- Specific legal citations IF present (Luật, Nghị định, Điều, Khoản...).
- Component terms that could be Entity names:
  * "công ty hợp danh", "chi nhánh", "doanh nghiệp tư nhân"
- These are used to search for ENTITIES in a knowledge graph.

Example thought process:
Query: "Hồ sơ đăng ký công ty hợp danh"
- This asks about REGISTRATION DOCUMENTS for a specific company type
- high_level: ["hồ sơ đăng ký công ty hợp danh", "hồ sơ đăng ký", "thủ tục đăng ký"]
- low_level: ["hồ sơ đăng ký công ty hợp danh", "công ty hợp danh"]

If the query contains no meaningful legal content, return empty arrays.

User Query: {query}
"""

PROMPTS["keywords_extraction_examples"] = [
    
    """Example 1: (Query contains explicit citation)
Query: "Theo Điều 7 Nghị định 01/2021/NĐ-CP, hồ sơ gồm những gì?"
Output:
{{
  "high_level_keywords": ["thành phần hồ sơ", "quy định pháp lý"],
  "low_level_keywords": ["Điều 7", "Nghị định 01/2021/NĐ-CP"]
}}
""",

    """Example 2: (Query about a SPECIFIC PROCEDURE - combine terms)
Query: "Hồ sơ đăng ký công ty hợp danh"
Output:
{{
  "high_level_keywords": [
    "hồ sơ đăng ký công ty hợp danh",
    "hồ sơ đăng ký",
    "thủ tục đăng ký"
  ],
  "low_level_keywords": [
    "hồ sơ đăng ký công ty hợp danh",
    "công ty hợp danh"
  ]
}}
Explanation: The FULL phrase "hồ sơ đăng ký công ty hợp danh" MUST appear in BOTH high_level and low_level to maximize search coverage.
""",

    """Example 3: (Query about a Specific Legal Form/Document)
Query: "Nội dung giấy đề nghị đăng ký doanh nghiệp"
Output:
{{
  "high_level_keywords": [
    "nội dung giấy đề nghị đăng ký doanh nghiệp",
    "nội dung giấy tờ",
    "thủ tục đăng ký"
  ],
  "low_level_keywords": [
    "nội dung giấy đề nghị đăng ký doanh nghiệp",
    "giấy đề nghị đăng ký doanh nghiệp"
  ]
}}
""",

    """Example 4: (Query about a Concept/Definition)
Query: "Thế nào là doanh nghiệp nhà nước?"
Output:
{{
  "high_level_keywords": [
    "định nghĩa",
    "khái niệm pháp lý",
    "phân loại doanh nghiệp"
  ],
  "low_level_keywords": [
    "doanh nghiệp nhà nước"
  ]
}}
""",

    """Example 5: (Query about CONDITIONS/REQUIREMENTS)
Query: "Điều kiện làm chủ tịch hội đồng quản trị"
Output:
{{
  "high_level_keywords": [
    "điều kiện làm chủ tịch hội đồng quản trị",
    "điều kiện",
    "tiêu chuẩn nhân sự"
  ],
  "low_level_keywords": [
    "điều kiện làm chủ tịch hội đồng quản trị",
    "chủ tịch hội đồng quản trị"
  ]
}}
""",

    """Example 6: (Query about ESTABLISHMENT PROCEDURE)
Query: "Thủ tục thành lập chi nhánh công ty"
Output:
{{
  "high_level_keywords": [
    "thủ tục thành lập chi nhánh công ty",
    "thủ tục thành lập",
    "đăng ký hoạt động"
  ],
  "low_level_keywords": [
    "thủ tục thành lập chi nhánh công ty",
    "chi nhánh công ty",
    "chi nhánh"
  ]
}}
""",

    """Example 7: (Query about DOCUMENT REQUIREMENTS for specific company type)
Query: "Hồ sơ thành lập công ty TNHH một thành viên"
Output:
{{
  "high_level_keywords": [
    "hồ sơ thành lập công ty TNHH một thành viên",
    "hồ sơ thành lập",
    "thủ tục thành lập"
  ],
  "low_level_keywords": [
    "hồ sơ thành lập công ty TNHH một thành viên",
    "công ty TNHH một thành viên"
  ]
}}
"""
]

# PROMPTS["keywords_extraction_examples"] = [
#     """Example 1:

# Query: "Theo Điều 7 Nghị định 01/2021/NĐ-CP, hồ sơ đăng ký doanh nghiệp gồm những giấy tờ gì?"

# Output:
# {{
#   "high_level_keywords": [
#     "hồ sơ đăng ký doanh nghiệp",
#     "quy định pháp lý",
#     "yêu cầu giấy tờ"
#   ],
#   "low_level_keywords": [
#     "Điều 7",
#     "Nghị định 01/2021/NĐ-CP",
#     "giấy tờ trong hồ sơ",
#     "đăng ký doanh nghiệp"
#   ]
# }}

# """,
#     """Example 2:

# Query: "Khoản 3 Điều 12 Luật Đầu tư 2020 quy định những hành vi bị cấm nào?"

# Output:
# {{
#   "high_level_keywords": [
#     "hành vi bị cấm",
#     "quy định pháp luật",
#     "đầu tư kinh doanh"
#   ],
#   "low_level_keywords": [
#     "Khoản 3",
#     "Điều 12",
#     "Luật Đầu tư 2020",
#     "hành vi bị cấm trong đầu tư"
#   ]
# }}

# """,
#     """Example 3:

# Query: "Điều 25 của Nghị định 122/2021/NĐ-CP có dẫn chiếu đến quy định nào tại Điều 21?"

# Output:
# {{
#   "high_level_keywords": [
#     "dẫn chiếu pháp lý",
#     "quy định liên quan",
#     "viện dẫn giữa các điều"
#   ],
#   "low_level_keywords": [
#     "Điều 25",
#     "Điều 21",
#     "Nghị định 122/2021/NĐ-CP",
#     "dẫn chiếu"
#   ]
# }}
# """
#     """Example 4:
# Query: "Điều kiện để thành lập văn phòng đại diện theo Luật Doanh nghiệp 2020 là gì?"

# Output:
# {{
#   "high_level_keywords": [
#     "điều kiện thành lập",
#     "thành lập văn phòng đại diện",
#     "thủ tục pháp lý"
#   ],
#   "low_level_keywords": [
#     "Luật Doanh nghiệp 2020",
#     "văn phòng đại diện",
#     "điều kiện thành lập"
#   ]
# }}
# """
#     """Example 5:
# Query: "Theo Điểm b Khoản 2 Điều 7 Luật Doanh nghiệp 2020, nghĩa vụ kê khai của doanh nghiệp là gì?"

# Output:
# {{
#   "high_level_keywords": [
#     "nghĩa vụ kê khai",
#     "trách nhiệm pháp lý",
#     "quy định nghĩa vụ doanh nghiệp"
#   ],
#   "low_level_keywords": [
#     "Điểm b",
#     "Khoản 2",
#     "Điều 7",
#     "Luật Doanh nghiệp 2020",
#     "nghĩa vụ kê khai"
#   ]
# }}
# """
#     """Example 6:
# Query: "Hồ sơ thay đổi người đứng đầu chi nhánh theo Nghị định 168/2025/NĐ-CP bao gồm những gì?"

# Output:
# {{
#   "high_level_keywords": [
#     "hồ sơ thay đổi",
#     "thủ tục hành chính",
#     "thay đổi người đứng đầu chi nhánh"
#   ],
#   "low_level_keywords": [
#     "Nghị định 168/2025/NĐ-CP",
#     "hồ sơ thay đổi người đứng đầu",
#     "chi nhánh"
#   ]
# }}
# """
#     """Example 7:
# Query: "Mức xử phạt theo Điều 21 Nghị định 122/2021/NĐ-CP đối với hành vi vi phạm kế toán là bao nhiêu?"

# Output:
# {{
#   "high_level_keywords": [
#     "mức xử phạt",
#     "vi phạm kế toán",
#     "quy định xử phạt"
#   ],
#   "low_level_keywords": [
#     "Điều 21",
#     "Nghị định 122/2021/NĐ-CP",
#     "hành vi vi phạm kế toán"
#   ]
# }}
# """,
# ]
