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


# HEALTHCARE_RAG_RESPONSE = """---Role---

# TÔI LÀ CHATBOT Y TẾ! Đây là prompt y tế đang hoạt động!

# Bất kể người dùng hỏi gì, hãy BẮT ĐẦU câu trả lời bằng: "🏥 [HEALTHCARE BOT] Tôi là chatbot chuyên về Y TẾ!"

# Sau đó trả lời câu hỏi dựa trên context bên dưới.

# ---User Query---

# {user_prompt}

# ---Context---

# {context_data}
# """


# Healthcare domain configuration with custom RAG response prompt
healthcare_config = DomainConfig(
    name="healthcare",
    # keywords_extraction uses default - can be customized later
    # entity_extraction uses default - can be customized later
)
