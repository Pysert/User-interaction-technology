import gradio as gr
import requests
import os
from typing import List, Tuple

# ================== 配置区 ==================
# DeepSeek 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-cb620f7166974722ada86223070218fc")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Kimi 配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "sk-VtldEiQVCq5LYLHqcbIiqXMVZB1JAxEK4NnTb8jEm7jzXErW")
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"


# ================== 模型核心逻辑 ==================
def build_messages(history: List[Tuple[str, str]], new_input: str) -> List[dict]:
    """统一消息格式构建"""
    messages = []
    for user, assistant in history:
        messages.extend([
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ])
    messages.append({"role": "user", "content": new_input})
    return messages


def call_deepseek(prompt: str, history: List[Tuple[str, str]]) -> str:
    """调用DeepSeek模型"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json={
                "model": "deepseek-chat",
                "messages": build_messages(history, prompt),
                "temperature": 0.3,
                "max_tokens": 2048
            },
            timeout=30
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"🔴 DeepSeek错误: {str(e)}"


def call_kimi(prompt: str, history: List[Tuple[str, str]]) -> str:
    """调用Kimi模型"""
    headers = {
        "Authorization": f"Bearer {KIMI_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            KIMI_API_URL,
            headers=headers,
            json={
                "model": "moonshot-v1-8k",
                "messages": build_messages(history, prompt),
                "temperature": 0.5,
                "max_tokens": 4096
            },
            timeout=40  # Kimi响应时间较长
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"🔴 Kimi错误: {str(e)}"


# ================== 主处理函数 ==================
def handle_conversation(selected_model, user_input, chat_history):
    try:
        if selected_model == "DeepSeek":
            reply = call_deepseek(user_input, chat_history)
        elif selected_model == "Kimi":
            reply = call_kimi(user_input, chat_history)

        chat_history.append((user_input, reply))
        return "", chat_history
    except Exception as e:
        return f"⚠️ 系统错误: {str(e)}", chat_history


def clear_chat():
    return []  # 返回空的字典列表，清空对话


# ================== Gradio界面 ==================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="orange",
    font=[gr.themes.GoogleFont("Noto Sans SC")]
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## 🌙 AI双模型对话系统（DeepSeek & Kimi）")

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["DeepSeek", "Kimi"],
            value="DeepSeek",
            label="选择对话模型",
            interactive=True
        )
        clear_btn = gr.Button("✨ 清空对话", variant="secondary")

    chatbot = gr.Chatbot(
        height=500,
        bubble_full_width=False,
        avatar_images=(
            "https://img2.baidu.com/it/u=3921464713,1750126262&fm=253&fmt=auto&app=138&f=PNG?w=500&h=500",  # 用户头像URL
            "https://p1.itc.cn/q_70/images03/20230908/8bb29620b4db40368ca362bd440b8412.png"  # 机器人头像URL
        )
    )

    msg_input = gr.Textbox(
        label="💬 输入消息",
        placeholder="请输入您的问题...",
        max_lines=5
    )

    # 交互绑定
    msg_input.submit(
        handle_conversation,
        [model_selector, msg_input, chatbot],
        [msg_input, chatbot]
    )
    clear_btn.click(clear_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # 设置为True可生成临时公网链接
    )



