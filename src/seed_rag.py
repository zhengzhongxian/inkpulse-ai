import asyncio
import logging
import numpy as np
import ollama
from redisvl.index import AsyncSearchIndex
from redisvl.schema import IndexSchema

logging.basicConfig(level=logging.INFO)

schema = IndexSchema.from_yaml("config/book_rag_schema.yaml")

SAMPLE_BOOKS = [
    {
        "id": "book_001",
        "title": "Nhà Giả Kim",
        "author": "Paulo Coelho",
        "category": "Văn học",
        "price": "79000",
        "content": "Nhà Giả Kim là tiểu thuyết bán chạy nhất thế giới. Hành trình của chàng chăn cừu Santiago đi tìm kho báu mang lại bài học sâu sắc về ước mơ và định mệnh."
    },
    {
        "id": "book_002",
        "title": "Đắc Nhân Tâm",
        "author": "Dale Carnegie",
        "category": "Kỹ năng sống",
        "price": "86000",
        "content": "Đắc Nhân Tâm là cuốn sách kỹ năng sống hot nhất và hay nhất mọi thời đại, giúp nâng cao nghệ thuật giao tiếp và ứng xử."
    },
    {
        "id": "book_003",
        "title": "Tuổi Trẻ Đáng Giá Bao Nhiêu",
        "author": "Rosie Nguyễn",
        "category": "Hạt giống tâm hồn",
        "price": "90000",
        "content": "Sách mới xuất bản truyền cảm hứng cho giới trẻ về học tập, đi và trải nghiệm những năm tháng thanh xuân quý giá."
    },
    {
        "id": "book_004",
        "title": "Tâm Lý Học Đám Đông",
        "author": "Gustave Le Bon",
        "category": "Tâm lý học",
        "price": "110000",
        "content": "Tác phẩm phân tích sâu sắc tâm lý xã hội và đám đông, nằm trong top các cuốn sách bán chạy nhất của nhà sách InkPulse."
    },
    {
        "id": "book_005",
        "title": "Khéo Ăn Khéo Nói Sẽ Có Được Thiên Hạ",
        "author": "Trác Nhã",
        "category": "Kỹ năng giao tiếp",
        "price": "95000",
        "content": "Sách hướng dẫn kỹ năng ăn nói khéo léo, nghệ thuật thuyết phục trong công việc và cuộc sống hàng ngày."
    },
    {
        "id": "book_006",
        "title": "Mật Mã Da Vinci",
        "author": "Dan Brown",
        "category": "Trinh thám",
        "price": "145000",
        "content": "Mật Mã Da Vinci là cuốn tiểu thuyết trinh thám hình sự giật gân hàng đầu thế giới của Dan Brown. Câu chuyện xoay quanh giáo sư Robert Langdon giải mã các biểu tượng bí ẩn và âm mưu tôn giáo kịch tính."
    }
]

async def seed():
    ollama_client = ollama.AsyncClient(host="http://192.168.80.1:11434")
    index = AsyncSearchIndex(schema, redis_url="redis://:RedisSecret123@redis-0.redis-headless.default.svc.cluster.local:6379")
    
    try:
        await index.create(overwrite=True)
        print("Created RAG Index schema!")
    except Exception as e:
        print(f"Index note: {e}")

    docs = []
    for b in SAMPLE_BOOKS:
        text = f"{b['title']} {b['author']} {b['category']} {b['content']}"
        print(f"Embedding: {b['title']}...")
        res = await ollama_client.embeddings(model="nomic-embed-text", prompt=text)
        b["vector"] = np.array(res["embedding"], dtype=np.float32).tobytes()
        docs.append(b)

    await index.load(docs)
    print("SUCCESS_LOADED_5_BOOKS")

if __name__ == "__main__":
    asyncio.run(seed())
