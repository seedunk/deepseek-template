import os
import json
import requests

# ========== 配置 ==========
API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-你的APIKey")
# API_URL = "https://api.deepseek.com/v1/chat/completions"
API_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "deepseek-v4-pro"

HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {API_KEY}",
}

# ========== 工具定义 ==========
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：杭州、北京、上海",
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式 YYYY-mm-dd",
                    },
                },
                "required": ["location", "date"],
            },
        },
    }
]


# ========== 模拟工具执行 ==========
def execute_tool(tool_call):
    """根据 tool_call 执行真实函数，这里用模拟数据代替"""
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    if name == "get_weather":
        location = args.get("location")
        date = args.get("date")
        # 模拟返回结果
        return f"{location}, {date}, 多云, 10~12℃"

    return f"未知工具: {name}"


# ========== 第一次请求：发送用户问题 + 工具定义 ==========
def first_call(user_message):
    payload = {
        "model": MODEL,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "messages": [
            {"role": "user", "content": user_message}
        ],
        "tools": tools,
    }

    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()

    assistant_msg = data["choices"][0]["message"]
    print("=== 第一次响应 ===")
    print("content:", assistant_msg.get("content"))
    print("reasoning_content:", assistant_msg.get("reasoning_content"))
    print("tool_calls:", json.dumps(assistant_msg.get("tool_calls"), ensure_ascii=False, indent=2))

    return assistant_msg
 

# ========== 第二次请求：回传 reasoning_content + 工具结果 ==========
def second_call(user_message, assistant_msg):
    # 1. 提取第一次响应中的关键字段
    content = assistant_msg.get("content") or ""          # 若为 null，改成空字符串
    reasoning_content = assistant_msg.get("reasoning_content")  # 必须原样回传
    tool_calls = assistant_msg.get("tool_calls")

    if not tool_calls:
        print("模型没有请求工具，直接返回内容：", content)
        return content

    # 2. 构造 messages 数组
    messages = [
        {"role": "user", "content": user_message},
        {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,   # 关键：必须携带
            "tool_calls": tool_calls,
        },
    ]

    # 3. 执行每一个 tool_call，并追加 tool 结果消息
    for tc in tool_calls:
        result = execute_tool(tc)
        print(f"执行工具 {tc['function']['name']} -> {result}")
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })

    # 4. 第二次请求
    payload = {
        "model": MODEL,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
        "messages": messages,
        "tools": tools,
    }

    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()

    final_msg = data["choices"][0]["message"]
    print("\n=== 第二次响应 ===")
    print("content:", final_msg.get("content"))
    print("reasoning_content:", final_msg.get("reasoning_content"))
    print("tool_calls:", final_msg.get("tool_calls"))

    return final_msg.get("content")


# ========== 主流程 ==========
if __name__ == "__main__":
    user_message = "杭州明天天气怎么样？"

    # 第一次轮询
    assistant_msg = first_call(user_message)

    print("---------------------------------------")
    
    print(assistant_msg)
    # 第二次轮询
    # final_answer = second_call(user_message, assistant_msg)

    # print("\n=== 最终回答 ===")
    # print(final_answer)