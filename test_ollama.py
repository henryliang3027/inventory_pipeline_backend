import ollama

# 系統提示詞
SYSTEM_PROMPT = """你是一位專業的超商貨架分析員。我會給你一組掃描結果清單，請根據清單回答用戶問題。如果清單中出現「未知商品」，請提醒用戶可能是新上架產品。

回答規則：
1. 根據關鍵字匹配商品名稱
2. 用繁體中文回答
3. 回答要簡潔明確
4. 如果找不到相關商品，請說明清單中沒有該商品"""

MODEL_NAME = "mistral:3b"


def ask_shelf_analyst(scan_results: dict, question: str) -> str:
    """
    使用 Ollama 模型回答貨架掃描問題

    Args:
        scan_results: 掃描結果字典 {商品名稱: 數量}
        question: 用戶問題

    Returns:
        模型回答
    """
    # 格式化掃描結果
    results_text = "\n".join([f"  {name}: {count} 瓶" for name, count in scan_results.items()])

    # 建立完整提示
    user_message = f"""掃描結果清單：
{results_text}

用戶問題：{question}"""

    # 呼叫 Ollama API
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    # 模擬掃描結果
    scan_results = {
        "LP33機能優酪乳": 2,
        "茶裏王半熟金萱": 1,
        "茶裏王白毫烏龍": 1
    }

    print("=" * 50)
    print("掃描結果清單：")
    print("=" * 50)
    for name, count in scan_results.items():
        print(f"  {name}: {count} 瓶")

    # 測試問題
    questions = [
        "有幾瓶優酪乳？",
        "茶裏王有幾瓶？",
        "有沒有可樂？",
        "貨架上總共有幾瓶飲料？"
    ]

    print("\n" + "=" * 50)
    print("問答測試：")
    print("=" * 50)

    for q in questions:
        print(f"\n問：{q}")
        answer = ask_shelf_analyst(scan_results, q)
        print(f"答：{answer}")
