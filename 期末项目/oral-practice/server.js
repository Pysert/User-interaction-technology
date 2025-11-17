import express from 'express';
import cors from 'cors';
import axios from 'axios';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// 中间件
app.use(cors());
app.use(express.json());

// DeepSeek API 配置
const DEEPSEEK_API_KEY = process.env.OPENAI_API_KEY;
const DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions';

// 系统 Prompt 配置
const SYSTEM_PROMPT = `You are an English speaking coach specialized in helping non-native speakers improve their oral English skills.

STRICT OUTPUT RULES:
1. Absolutely NO decorative symbols like *, ~, -, _ or any other special characters
2. No bullet points or numbered lists - use full sentences only
3. No markdown or formatting syntax
4. Only use standard punctuation: , . ? !
5. Responses must be 100% plain text

Teaching methodology:
1. Gently correct errors by: "You said: [quote]. A more natural way is: [correction]"
2. Always ask an open-ended follow-up question
3. Keep responses between 100-200 words
4. Use simple, clear English at CEFR B1 level`;

// 聊天接口
app.post('/chat', async (req, res) => {
  try {
    const { messages, model = 'deepseek-chat', max_tokens = 1000, temperature = 0.7 } = req.body;

    if (!DEEPSEEK_API_KEY) {
      return res.status(500).json({ error: 'API key not configured' });
    }

    // 构建消息数组，系统消息在最前面
    const processedMessages = [
      {
        role: 'system',
        content: SYSTEM_PROMPT
      },
      ...messages
    ];

    // 调用 DeepSeek API
    const response = await axios.post(
      DEEPSEEK_API_URL,
      {
        model,
        messages: processedMessages,
        max_tokens,
        temperature,
        frequency_penalty: 0.5,
        presence_penalty: 0.5
      },
      {
        headers: {
          'Authorization': `Bearer ${DEEPSEEK_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    // 过滤响应中的特殊符号
    const filteredContent = filterSymbols(response.data.choices[0].message.content);

    // 记录包含符号的响应（用于调试）
    if (response.data.choices[0].message.content !== filteredContent) {
      console.warn('Filtered symbols from response:', {
        original: response.data.choices[0].message.content,
        filtered: filteredContent
      });
    }

    // 返回过滤后的响应
    res.json({
      ...response.data,
      choices: [{
        ...response.data.choices[0],
        message: {
          ...response.data.choices[0].message,
          content: filteredContent
        }
      }]
    });

  } catch (error) {
    console.error('DeepSeek API Error:', {
      message: error.message,
      response: error.response?.data,
      stack: error.stack
    });
    
    res.status(500).json({ 
      error: 'Failed to get response from API',
      details: error.response?.data || error.message,
      type: 'api_error'
    });
  }
});

// 符号过滤函数
function filterSymbols(text) {
  return text
    .replace(/[*_~-]/g, '')         // 移除装饰符号
    .replace(/^\s*\d+\.\s*/gm, '')  // 移除编号列表
    .replace(/\n\s*\n/g, '\n')      // 压缩多余空行
    .trim();
}

// 健康检查
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    message: 'Server is running',
    api: 'DeepSeek',
    prompt_version: '1.2' 
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`DeepSeek API configured: ${DEEPSEEK_API_URL}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
});