# Frontend API Guide

## Request
http://101.101.96.40:9621/query/stream

```javascript
POST /query/stream
{
  "query": "muốn đuổi giám đốc thì làm sao",
  "mode": "mix",
  "stream": true,
  "include_references": true,
  "include_chunk_content": true
}
```

## Response

Response là **NDJSON** (mỗi dòng là 1 JSON object):

### Dòng 1: References (danh sách điều luật)

```json
{
  "references": [
    {
      "reference_id": "1",
      "file_path": "Luật_Phá_sản_2014.txt",
      "content": ["Điều 54. Thứ tự phân chia tài sản\n1. Khi tuyên bố phá sản..."]
    },
    {
      "reference_id": "2",
      "file_path": "Luật_Phá_sản_2014.txt", 
      "content": ["Điều 108. Quyết định tuyên bố phá sản\n1. Quyết định phải có..."]
    }
  ]
}
```

| Field          | Mô tả                                                              |
| -------------- | ------------------------------------------------------------------ |
| `reference_id` | ID dùng để map với citation `[1]`, `[2]` trong response            |
| `file_path`    | Tên file nguồn                                                     |
| `content`      | **Array** chứa nội dung điều luật (dùng `.join('\n')` để hiển thị) |

### Dòng 2+: Response (streaming)

```json
{"response": "### Kết luận\nTheo Điều 54 ([1])..."}
{"response": "\n\n### Căn cứ pháp lý\n- Điều 54 ([1])"}
```

Response chứa citations dạng `[1]`, `[2]` → map với `reference_id` để hiện popup.

## Popup điều luật

```javascript
let refs = [];  // Lưu từ dòng 1

function showPopup(id) {
  const ref = refs.find(r => r.reference_id === id);
  showModal({
    title: ref.file_path,
    body: ref.content.join('\n')
  });
}
```

## Lưu ý

- Mỗi điều luật có ID riêng (Điều 54 = `[1]`, Điều 108 = `[2]`)
- Response có thể chứa link download: `[Mẫu 1](https://...)`
