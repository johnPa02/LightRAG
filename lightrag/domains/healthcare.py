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


# Healthcare domain configuration with custom keywords extraction prompt
healthcare_config = DomainConfig(
    name="healthcare",
    keywords_extraction=HEALTHCARE_KEYWORDS_EXTRACTION,
    # entity_extraction uses default - can be customized later
)
