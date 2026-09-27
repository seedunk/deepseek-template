import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import json
import uuid
import time
import re

# 环境变量设置（解决ConstTensorWrapper问题）
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 禁用torch._dynamo
import torch
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True

# 初始化FastAPI应用
app = FastAPI(title="DeepSeek R1 API", version="1.0.0")

# 全局模型变量
_model = None
_tokenizer = None
_MODEL_PATH =  os.getenv("deepseek_r1_distill_qwen_7b", "../models/deepseek_r1_distill_qwen_7b")   

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# ============ Pydantic模型定义============
# 复刻Deepseek官网API

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class Tool(BaseModel):
    type: str = "function"
    function: ToolFunction

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    tools: Optional[List[Tool]] = None  # 从前端获取的工具定义
    tool_choice: Optional[str] = "auto"  # auto, none, required

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

# ============ 模型加载 ============

def get_model():
    """获取模型（只加载一次）"""
    global _model, _tokenizer
    if _model is None:
        print("🔄 正在加载模型...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_PATH,
                trust_remote_code=True
            )
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            _model = AutoModelForCausalLM.from_pretrained(
                _MODEL_PATH,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            _model.eval()
            print(f"✅ 模型加载完成！设备: {_model.device}")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise
    return _tokenizer, _model

# ============ 提示词构建 ============

def build_tools_prompt(tools: List[Tool]) -> str:
    if not tools:
        return ""
    
    tools_desc = []
    for tool in tools:
        if tool.type == "function":
            func = tool.function
            tools_desc.append(f"- {func.name}: {func.description or '无描述'}")
            if func.parameters:
                tools_desc.append(f"  参数: {json.dumps(func.parameters, ensure_ascii=False)}")
    
    return f"""
你拥有以下工具，但仅在用户明确需要实时数据、外部信息或无法从自身知识回答时才调用**。对于一般性问题，请直接回答，不要使用工具。

可用工具：
{chr(10).join(tools_desc)}

调用格式（仅在必要时使用）：
<tool_call>{{"name": "工具名称", "arguments": {{"参数名": "参数值"}}}}</tool_call>

例如：
- 用户问“北京天气如何？” → 你需要调用 get_weather。 

请根据用户问题自主判断，不要强行调用工具。
"""


def build_chat_prompt(messages: List[Message], tools: Optional[List[Tool]] = None) -> str:
    """构建完整的对话提示词"""
    prompt_parts = []
    
    # 添加系统提示
    prompt_parts.append("<|system|>\n你是一个有帮助的AI助手。你可以利用提供的工具来回答特定问题，但不要过度使用。")
    
    # 添加工具说明（如果有）
    if tools:
        tools_prompt = build_tools_prompt(tools)
        if tools_prompt:
            prompt_parts.append(f"\n{tools_prompt}")
    
    # 添加历史消息
    for msg in messages:
        if msg.role == "system":
            prompt_parts.append(f"<|system|>\n{msg.content}")
        elif msg.role == "user":
            prompt_parts.append(f"<|user|>\n{msg.content}")
        elif msg.role == "assistant":
            if msg.tool_calls:
                # 处理工具调用消息
                for call in msg.tool_calls:
                    if call.get("function"):
                        prompt_parts.append(
                            f"<|assistant|>\n[调用工具: {call['function'].get('name', '')}]"
                        )
            else:
                prompt_parts.append(f"<|assistant|>\n{msg.content}")
        elif msg.role == "tool":
            prompt_parts.append(f"<|tool|>\n{msg.content}")
    
    prompt_parts.append("<|assistant|>\n")
    return "\n".join(prompt_parts)

# ============ 工具调用解析 ============

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """从模型输出中解析工具调用"""
    tool_calls = []
    
    # 解析 <tool_call> 标签
    pattern = r'<tool_call>(.*?)</tool_call>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            data = json.loads(match.strip())
            if "name" in data:
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": data["name"],
                        "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)
                    }
                })
        except json.JSONDecodeError:
            continue
    
    # 如果没有找到标签，尝试直接解析JSON
    if not tool_calls:
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\s*\}'
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                data = json.loads(match)
                if "name" in data:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:12]}",
                        "type": "function",
                        "function": {
                            "name": data["name"],
                            "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)
                        }
                    })
            except json.JSONDecodeError:
                continue
    
    return tool_calls

def clean_response_text(text: str) -> str:
    """清理响应文本，移除工具调用标记"""
    # 移除 <tool_call> 标签
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # 清理多余的空行
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()

# ============ FastAPI端点 ============

@app.on_event("startup")
async def startup_event():
    """服务启动时预加载模型"""
    try:
        get_model()
        print("🚀 服务已启动并加载模型")
    except Exception as e:
        print(f"⚠️ 启动时模型加载失败: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "DeepSeek API 服务",
        "status": "running",
        "endpoints": {
            "/v1/chat/completions": "POST - 聊天补全接口（复刻官网）",
            "/v1/models": "GET - 模型列表",
            "/health": "GET - 健康检查"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查接口"""
    try:
        tokenizer, model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(model.device) if model else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/v1/models")
async def list_models():
    """列出可用模型（复刻官网格式）"""
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-chat",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek"
            },
            {
                "id": "deepseek-reasoner",
                "object": "model",
                "created": 1700000000,
                "owned_by": "deepseek"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天补全接口 - 完全复刻DeepSeek官网API格式
    工具定义从前端获取，服务端不执行工具
    """
    try:
        tokenizer, model = get_model()
        
        # 1. 构建提示词
        prompt = build_chat_prompt(request.messages, request.tools)
        
        # 2. 生成响应
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens or 2048,
                do_sample=True,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 3. 解码响应
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取新生成的部分
        response_text = full_response[len(prompt):] if full_response.startswith(prompt) else full_response
        
        # 4. 解析工具调用
        tool_calls = parse_tool_calls(response_text)
        
        # 5. 清理响应文本（移除工具调用标记）
        clean_response = clean_response_text(response_text)
        
        # 6. 构建消息
        message = {
            "role": "assistant",
            "content": clean_response if clean_response else None
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
            # 如果没有文本内容，content设为None
            if not clean_response:
                message["content"] = None
        
        # 7. 构建响应（完全符合官网格式）
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created_time = int(time.time())
        
        # 计算token数（简化计算）
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(response_text) // 4
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created_time,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============ 启动服务 ============

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 DeepSeek API 服务 (完全复刻官网格式)")
    print("=" * 60)
    print(f"📍 服务地址: http://0.0.0.0:8000")
    print(f"📚 API 文档: http://0.0.0.0:8000/docs")
    print(f"❤️  健康检查: http://0.0.0.0:8000/health")
    print("=" * 60)
    print("工具配置: 从前端请求获取，服务端不执行")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    
    uvicorn.run(
        "deepseek_api:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
    )