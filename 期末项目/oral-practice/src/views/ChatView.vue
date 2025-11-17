<template>
  <div class="chat-layout">
    <div class="chat-container">
      <div class="chat-header">
        <h1>English Speaking Practice</h1>
        <p>Click the record button to start practicing English</p>     
          <div class="api-status" :class="{ 'demo-mode': isDemoMode }">
          {{ isDemoMode ? 'Demo Mode (Using Sample Data)' : 'Normal Mode (Using DeepSeek API)' }}
          </div>
      </div>

      <div class="messages-container" ref="messagesContainer">
        <div 
          v-for="(message, index) in messages" 
          :key="index" 
          :class="['message', message.role === 'user' ? 'user-message' : 'assistant-message']"
        >
          <div class="message-content">
            <div class="message-header" @click="toggleMessageVisibility(index)">
              <div class="message-preview">
                <span class="role-indicator">{{ message.role === 'user' ? 'You' : 'Assistant' }}</span>
                <span class="preview-text">{{ message.isExpanded ? 'Click to hide' : 'Click to show text' }}</span>
              </div>
              <div class="expand-icon" :class="{ 'expanded': message.isExpanded }">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M7 10l5 5 5-5z"/>
                </svg>
              </div>
            </div>
            <div 
              v-show="message.isExpanded" 
              class="message-text"
              :class="{ 'fade-in': message.isExpanded }"
            >
              {{ message.content }}
            </div>
            <div class="message-footer">
              <div class="message-time">{{ formatTime(message.timestamp) }}</div>
              <button 
                v-if="message.role === 'assistant'"
                @click.stop="playMessage(message)"
                :disabled="isPlaying"
                class="play-button"
                :class="{ 'playing': isPlaying && currentPlayingIndex === index }"
              >
                <svg v-if="!(isPlaying && currentPlayingIndex === index)" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z"/>
                </svg>
                <svg v-else width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      <div class="input-container">
        <div class="voice-input-area">
          <div class="status-text">{{ statusText }}</div>
          <div class="button-group">
            <div class="main-buttons">
              <button 
                @click="toggleRecording" 
                :class="['record-button', { 'recording': isRecording }]"
                :disabled="isProcessing"
              >
                <div class="record-icon">
                  <svg v-if="!isRecording" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                  </svg>
                  <div v-else class="recording-animation"></div>
                </div>
                {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
              </button>
              
              <button 
                v-if="isPlaying"
                @click="stopAllSpeech"
                class="stop-button"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 6h12v12H6z"/>
                </svg>
                Stop Audio
              </button>
            </div>
            
            <button @click="globalReset" class="global-reset" title="Global Reset"></button>
          </div>
        </div>
      </div>
    </div>

 
    <!-- 统计表 -->
    <div class="stats-panel">
      <div class="stats-header">
        <h3>Practice Statistics</h3>
        <div class="stats-header-controls">
          <button @click="resetStats" class="reset-button">Reset</button>
          <button 
            @click="statsPanelExpanded = !statsPanelExpanded" 
            class="toggle-button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M7 10l5 5 5-5z" 
                    :transform="statsPanelExpanded ? 'rotate(180 12 12)' : ''" />
            </svg>
          </button>
        </div>
      </div>
      
      <div v-show="statsPanelExpanded" class="stats-content">
        <div class="stat-item">
          <div class="stat-label">Text Views</div>
          <div class="stat-value">{{ computedStats.textViewCount }}</div>
          <div class="stat-description">Times you viewed assistant text</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Average Speed</div>
          <div class="stat-value">{{ computedStats.averageSpeed }} WPM</div>
          <div class="stat-description">Words per minute</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Total Messages</div>
          <div class="stat-value">{{ computedStats.totalMessages }}</div>
          <div class="stat-description">Conversation turns</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Assistant Messages</div>
          <div class="stat-value">{{ assistantMessageCount }}</div>
          <div class="stat-description">Messages from assistant</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Practice Time</div>
          <div class="stat-value">{{ computedStats.practiceTime }}</div>
          <div class="stat-description">Total practice duration</div>
        </div>
        
        <div class="stat-item">
          <div class="stat-label">Play Count</div>
          <div class="stat-value">{{ computedStats.playCount }}</div>
          <div class="stat-description">Times you replayed audio</div>
        </div>
        
        <div class="stat-item progress-item">
          <div class="stat-label">Text View Rate</div>
          <div class="progress-bar">
            <div 
              class="progress-fill" 
              :style="{ width: textViewRate + '%' }"
            ></div>
          </div>
          <div class="progress-text">{{ textViewRate }}%</div>
        </div>
      </div>
    </div>

    <!-- 设置栏 -->
    <div class="settings-panel">
      <div class="settings-header">
        <h3>Settings</h3>
        <button @click="toggleSettings" class="toggle-button">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M7 10l5 5 5-5z"/>
          </svg>
        </button>
      </div>
      
      <div v-show="showSettings" class="settings-content">
        <div class="settings-row">
          <div class="setting-item">
            <label class="setting-label">Speech Rate</label>
            <div class="setting-control">
              <input 
                type="range" 
                v-model="speechRate" 
                min="0.5" 
                max="2.0" 
                step="0.1"
                class="slider"
              >
              <span class="setting-value">{{ speechRate }}x</span>
            </div>
            <div class="setting-description">Adjust the speed of voice playback</div>
          </div>
          
          <div class="setting-item">
            <label class="setting-label">Voice Pitch</label>
            <div class="setting-control">
              <input 
                type="range" 
                v-model="voicePitch" 
                min="0.5" 
                max="2.0" 
                step="0.1"
                class="slider"
              >
              <span class="setting-value">{{ voicePitch }}x</span>
            </div>
            <div class="setting-description">Adjust the pitch of the voice</div>
          </div>
        </div>
        
        <div class="setting-item">
          <label class="setting-label">Auto Play</label>
          <div class="setting-control">
            <label class="toggle-switch">
              <input type="checkbox" v-model="autoPlay">
              <span class="toggle-slider"></span>
            </label>
          </div>
          <div class="setting-description">Automatically play assistant responses</div>
        </div>
        
        <div class="setting-item" v-if="availableVoices.length > 0">
          <label class="setting-label">Voice</label>
          <div class="setting-control">
            <select v-model="selectedVoice" class="voice-select">
              <option value="">Default Voice</option>
              <option 
                v-for="voice in availableVoices" 
                :key="voice.name" 
                :value="voice.name"
              >
                {{ voice.name }} ({{ voice.lang }})
              </option>
            </select>
          </div>
          <div class="setting-description">Choose your preferred voice</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch, computed, onUnmounted } from 'vue'

// 响应式数据
const messages = ref([])
const isRecording = ref(false)
const isProcessing = ref(false)
const statusText = ref('Click the record button to start practicing')
const messagesContainer = ref(null)
const isDemoMode = ref(false)
const isPlaying = ref(false)
const currentPlayingIndex = ref(-1)
const statsPanelExpanded = ref(true); // 控制统计面板的展开状态

// 设置相关
const showSettings = ref(true)
const speechRate = ref(0.9)
const voicePitch = ref(1.0)
const autoPlay = ref(true)
const selectedVoice = ref('')
const availableVoices = ref([])

// 统计数据
const stats = ref({
  textViewCount: 0,
  totalMessages: 0,
  totalWords: 0,
  totalDuration: 0, // 总录音时长（秒）
  practiceStartTime: null,
  lastRecordingStart: null,
  playCount: 0 // 语音播放次数
})

// 语音识别相关
let recognition = null
let synthesis = null

// 计算属性
const computedStats = computed(() => {
  const averageSpeed = stats.value.totalDuration > 0 
    ? Math.round((stats.value.totalWords / stats.value.totalDuration) * 60)
    : 0
  
  const practiceTime = stats.value.practiceStartTime 
    ? formatDuration(Date.now() - stats.value.practiceStartTime)
    : '0:00'
  
  return {
    textViewCount: stats.value.textViewCount,
    averageSpeed,
    totalMessages: stats.value.totalMessages,
    practiceTime,
    playCount: stats.value.playCount
  }
})

const textViewRate = computed(() => {
  // 只计算助手消息的数量
  const assistantMessages = messages.value.filter(msg => msg.role === 'assistant').length
  if (assistantMessages === 0) return 0
  return Math.round((stats.value.textViewCount / assistantMessages) * 100)
})

const assistantMessageCount = computed(() => {
  return messages.value.filter(msg => msg.role === 'assistant').length
})

// 样例对话数据
const demoResponses = {
  'hello': 'Hello! Nice to meet you! I\'m your English speaking practice assistant. How can I help you today?',
  'how are you': 'I\'m doing great, thank you for asking! How about you? I hope you\'re having a wonderful day.',
  'what is your name': 'My name is Alex, and I\'m here to help you practice English speaking. What\'s your name?',
  'tell me a joke': 'Here\'s a funny one: Why don\'t scientists trust atoms? Because they make up everything!',
  'what is artificial intelligence': 'Artificial Intelligence, or AI, is a branch of computer science that aims to create systems capable of performing tasks that typically require human intelligence. These include learning, reasoning, problem-solving, perception, and language understanding.',
  'recommend a movie': 'I\'d recommend "The Shawshank Redemption" - it\'s a classic film about hope and perseverance. Or "Forrest Gump" - a heartwarming story that\'s both funny and touching.',
  'how to learn english': 'Here are some tips to improve your English: 1. Practice speaking regularly, even if it\'s just talking to yourself. 2. Watch English movies and TV shows with subtitles. 3. Listen to English podcasts and music. 4. Read English books and articles. 5. Find a language exchange partner. 6. Don\'t be afraid to make mistakes - they\'re part of learning!',
  'thank you': 'You\'re very welcome! I\'m glad I could help. Feel free to ask me anything else!',
  'goodbye': 'Goodbye! I hope our conversation helped you practice English. Have a wonderful day!',
  'what time is it': 'I can\'t tell you the exact time, but I can help you practice asking about time in English. You could say "What time is it?" or "Could you tell me the time, please?"',
  'what is the weather like': 'I can\'t check the real-time weather, but I can help you practice weather-related vocabulary! You could say "It\'s sunny today" or "It looks like it might rain."',
  'default': 'That\'s an interesting question! Let me think about it... I believe there are many ways to look at this. What are your thoughts on this topic?'
}

// 初始化语音识别
onMounted(() => {
  initSpeechRecognition()
  initSpeechSynthesis()
  checkApiStatus()
  initStats()
  
  // 添加欢迎消息
  const welcomeMessage = 'Hello! I\'m your English speaking practice assistant. Click the record button to start practicing English with me!'
  messages.value.push({
    role: 'assistant',
    content: welcomeMessage,
    timestamp: new Date(),
    isExpanded: true // 默认展开
  })
  
  // 添加键盘事件监听
  document.addEventListener('keydown', handleKeyDown)
})

// 初始化统计数据
const initStats = () => {
  stats.value.practiceStartTime = Date.now()
  stats.value.totalMessages = 1 // 包含欢迎消息
}

// 重置统计数据
const resetStats = () => {
  stats.value = {
    textViewCount: 0,
    totalMessages: 1, // 保留欢迎消息
    totalWords: 0,
    totalDuration: 0,
    practiceStartTime: Date.now(),
    lastRecordingStart: null,
    playCount: 0
  }
}

// 检查API状态
const checkApiStatus = async () => {
  try {
    const response = await fetch('/api/health', { 
      method: 'GET',
      signal: AbortSignal.timeout(3000) // 3秒超时
    })
    if (response.ok) {
      isDemoMode.value = false
    } else {
      isDemoMode.value = true
    }
  } catch (error) {
    console.log('API unavailable, using demo mode')
    isDemoMode.value = true
  }
}

// 初始化语音识别
const initSpeechRecognition = () => {
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    recognition = new SpeechRecognition()
    
    recognition.continuous = false // 恢复为非连续模式，但延长停顿时间
    recognition.interimResults = true // 启用中间结果
    recognition.lang = 'en-US' // 改为英语
    
    // 设置更长的静音检测时间
    recognition.maxAlternatives = 1
    
    recognition.onstart = () => {
      isRecording.value = true
      stats.value.lastRecordingStart = Date.now()
      statusText.value = 'Recording... Speak naturally, it will stop after 3 seconds of silence'
    }
    
    recognition.onresult = (event) => {
      let finalTranscript = ''
      let interimTranscript = ''
      
      // 处理所有结果
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript
        if (event.results[i].isFinal) {
          finalTranscript += transcript
        } else {
          interimTranscript += transcript
        }
      }
      
      // 如果有最终结果，处理它
      if (finalTranscript) {
        statusText.value = 'Processing your speech...'
        handleUserInput(finalTranscript.trim())
      } else if (interimTranscript) {
        // 显示中间结果，让用户知道系统在听
        statusText.value = `Listening: "${interimTranscript}"`
      }
    }
    
    recognition.onerror = (event) => {
      console.error('Speech recognition error:', event.error)
      if (event.error === 'no-speech') {
        statusText.value = 'No speech detected, please try again'
      } else if (event.error === 'audio-capture') {
        statusText.value = 'Microphone not found, please check your device'
      } else if (event.error === 'not-allowed') {
        statusText.value = 'Microphone access denied, please allow microphone access'
      } else {
        statusText.value = 'Recognition failed, please try again'
      }
      isRecording.value = false
    }
    
    recognition.onend = () => {
      isRecording.value = false
      if (!isProcessing.value) {
        statusText.value = 'Click the record button to start practicing'
      }
    }
    
    // 添加自定义的静音检测，延长停顿时间
    recognition.onspeechend = () => {
      // 当检测到说话结束时，等待更长时间再停止
      setTimeout(() => {
        if (isRecording.value) {
          recognition.stop()
        }
      }, 3000) // 等待3秒静音后停止（原来是默认的较短时间）
    }
  } else {
    statusText.value = 'Your browser does not support speech recognition'
  }
}

// 初始化语音合成
const initSpeechSynthesis = () => {
  if ('speechSynthesis' in window) {
    synthesis = window.speechSynthesis
    
    // 加载可用的人声
    const loadVoices = () => {
      const voices = synthesis.getVoices()
      // 过滤出英语人声
      const englishVoices = voices.filter(voice => 
        voice.lang.startsWith('en') && !voice.localService
      )
      availableVoices.value = englishVoices
      
      // 如果没有选择人声，默认选择第一个英语人声
      if (!selectedVoice.value && englishVoices.length > 0) {
        selectedVoice.value = englishVoices[0].name
      }
      // 自动播放欢迎语（只执行一次）
      if (autoPlay.value && !initSpeechSynthesis.welcomePlayed) {
        const welcomeMessage = 'Hello! I\'m your English speaking practice assistant. Click the record button to start practicing English with me!'
        speakText(welcomeMessage)
        initSpeechSynthesis.welcomePlayed = true
      }
    }
    
    // 如果人声已经加载
    if (synthesis.getVoices().length > 0) {
      loadVoices()
    } else {
      // 等待人声加载完成
      synthesis.onvoiceschanged = loadVoices
      // fallback: 轮询，防止部分浏览器onvoiceschanged不触发
      let pollCount = 0
      const pollVoices = () => {
        if (synthesis.getVoices().length > 0) {
          loadVoices()
        } else if (pollCount < 20) {
          pollCount++
          setTimeout(pollVoices, 200)
        }
      }
      pollVoices()
    }
  }
}
initSpeechSynthesis.welcomePlayed = false;

// 切换录音状态
const toggleRecording = () => {
  if (isRecording.value) {
    // 手动停止录音
    recognition.stop()
    statusText.value = 'Stopping recording...'
  } else {
    // 开始录音
    try {
      recognition.start()
    } catch (error) {
      console.error('Failed to start recording:', error)
      statusText.value = 'Failed to start recording, please try again'
    }
  }
}

// 处理用户输入
const handleUserInput = async (text) => {
  isProcessing.value = true
  
  // 更新统计数据
  if (stats.value.lastRecordingStart) {
    const duration = (Date.now() - stats.value.lastRecordingStart) / 1000
    stats.value.totalDuration += duration
  }
  
  const wordCount = text.split(/\s+/).length
  stats.value.totalWords += wordCount
  stats.value.totalMessages += 1
  
  // 添加用户消息
  messages.value.push({
    role: 'user',
    content: text,
    timestamp: new Date(),
    isExpanded: false // 默认不展开
  })
  
  try {
    let response
    if (isDemoMode.value) {
      response = getDemoResponse(text)
      await new Promise(resolve => setTimeout(resolve, 1000))
    } else {
      response = await callDeepSeek(text) // 改为调用 callDeepSeek
    }
    
    // 添加助手回复
    messages.value.push({
      role: 'assistant',
      content: response,
      timestamp: new Date(),
      isExpanded: false // 默认不展开
    })
    
    stats.value.totalMessages += 1
    
    // 根据设置决定是否自动播放
    if (autoPlay.value) {
      speakText(response)
    }
    
    statusText.value = 'Click the record button to continue practicing'
  } catch (error) {
    console.error('Processing failed:', error)
    messages.value.push({
      role: 'assistant',
      content: 'Sorry, I encountered an issue. Please try again.',
      timestamp: new Date(),
      isExpanded: false // 默认不展开
    })
    stats.value.totalMessages += 1
    statusText.value = 'An error occurred, please try again'
  } finally {
    isProcessing.value = false
  }
}

// 切换消息可见性
const toggleMessageVisibility = (index) => {
  const message = messages.value[index]
  const wasExpanded = message.isExpanded
  
  message.isExpanded = !message.isExpanded
  
  // 只对助手消息计数，且只在从收起状态变为展开状态时计数
  if (message.role === 'assistant' && !wasExpanded && message.isExpanded) {
    stats.value.textViewCount += 1
  }
}

// 获取样例回复
const getDemoResponse = (text) => {
  // 简单的关键词匹配
  const lowerText = text.toLowerCase()
  
  for (const [key, value] of Object.entries(demoResponses)) {
    if (lowerText.includes(key.toLowerCase())) {
      return value
    }
  }
  
  return demoResponses['default']
}

// 调用 DeepSeek API
const callDeepSeek = async (message) => {
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        messages: [
          {
            role: 'system',
            content: 'You are an English speaking practice assistant. Help users practice English by having natural conversations. Keep responses conversational and encouraging. Correct any obvious grammar mistakes gently and provide helpful feedback.'
          },
          {
            role: 'user',
            content: message
          }
        ],
        model: 'deepseek-chat', // 使用 deepseek-chat 模型
        max_tokens: 1000,
        temperature: 0.7
      })
    })
    
    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.details || `HTTP error! status: ${response.status}`)
    }
    
    const data = await response.json()
    return data.choices[0].message.content
  } catch (error) {
    console.error('API call failed:', error)
    throw new Error(`API request failed: ${error.message}`)
  }
}
// 语音播放
const speakText = (text) => {
  if (!synthesis) {
    initSpeechSynthesis()
  }

  if (synthesis) {
    // 停止之前的播放
    synthesis.cancel()
    
    const utterance = new SpeechSynthesisUtterance(text)
    applySpeechSettings(utterance)

    utterance.onstart = () => {
      console.log('开始播放语音')
      statusText.value = 'Playing audio... Press Space to stop'
    }

    utterance.onend = () => {
      console.log('播放结束')
      isPlaying.value = false
      currentPlayingIndex.value = -1
      statusText.value = 'Click the record button to start practicing'
    }

    utterance.onerror = (event) => {
      console.error('语音播放错误:', event)
      isPlaying.value = false
      currentPlayingIndex.value = -1
      statusText.value = 'Audio playback error, please try again'
    }

    synthesis.speak(utterance)
  }
}

// 格式化时间
const formatTime = (timestamp) => {
  return timestamp.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 格式化持续时间
const formatDuration = (milliseconds) => {
  const totalSeconds = Math.floor(milliseconds / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

// 监听消息变化，自动滚动到底部
watch(messages, () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}, { deep: true })

// 播放消息语音
const playMessage = (message) => {
  if (isPlaying.value) {
    // 如果正在播放，停止播放
    if (synthesis) {
      synthesis.cancel()
    }
    isPlaying.value = false
    currentPlayingIndex.value = -1
    return
  }

  if (!synthesis) {
    initSpeechSynthesis()
  }

  if (synthesis) {
    // 增加播放次数统计
    stats.value.playCount++
    
    const utterance = new SpeechSynthesisUtterance(message.content)
    applySpeechSettings(utterance)

    // 设置播放状态
    isPlaying.value = true
    currentPlayingIndex.value = messages.value.findIndex(m => m === message)

    utterance.onstart = () => {
      console.log('开始播放语音')
    }

    utterance.onend = () => {
      console.log('播放结束')
      isPlaying.value = false
      currentPlayingIndex.value = -1
    }

    utterance.onerror = (event) => {
      console.error('语音播放错误:', event)
      isPlaying.value = false
      currentPlayingIndex.value = -1
    }

    synthesis.speak(utterance)
  }
}

// 停止所有语音播放
const stopAllSpeech = () => {
  if (synthesis) {
    synthesis.cancel()
  }
  isPlaying.value = false
  currentPlayingIndex.value = -1
  statusText.value = 'Audio stopped. Click the record button to start practicing'
}

// 切换设置面板显示
const toggleSettings = () => {
  showSettings.value = !showSettings.value
}

// 应用语音设置
const applySpeechSettings = (utterance) => {
  utterance.rate = speechRate.value
  utterance.pitch = voicePitch.value
  utterance.volume = 1.0
  utterance.lang = 'en-US'
  
  // 应用选择的人声
  if (selectedVoice.value) {
    const voices = synthesis.getVoices()
    const selectedVoiceObj = voices.find(voice => voice.name === selectedVoice.value)
    if (selectedVoiceObj) {
      utterance.voice = selectedVoiceObj
    }
  }
}

// 键盘事件处理
const handleKeyDown = (event) => {
  // 按空格键停止语音播放
  if (event.code === 'Space' && isPlaying.value) {
    event.preventDefault()
    stopAllSpeech()
  }
}

// 组件卸载时清理语音播放状态
onUnmounted(() => {
  stopAllSpeech()
  document.removeEventListener('keydown', handleKeyDown)
})

// 全局重置
const globalReset = () => {
  // 重置统计数据
  resetStats()
  
  // 清空消息
  messages.value = []
  
  // 重新添加欢迎消息
  const welcomeMessage = 'Hello! I\'m your English speaking practice assistant. Click the record button to start practicing English with me!'
  messages.value.push({
    role: 'assistant',
    content: welcomeMessage,
    timestamp: new Date(),
    isExpanded: true // 默认展开
  })
  
  // 重置语音播放状态
  stopAllSpeech()
  
  // 重置设置面板状态
  initSpeechSynthesis.welcomePlayed = false
  
  // 更新状态文本
  statusText.value = 'Click the record button to start practicing'
  
  // 如果自动播放开启，播放欢迎语
  if (autoPlay.value) {
    setTimeout(() => {
      speakText(welcomeMessage)
    }, 500)
  }
}
</script>

<style scoped>
.chat-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  background: #f5f5f5;
  overflow: hidden;
  position: relative;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f5f5f5;
  border-bottom: 1px solid #e0e0e0;
  min-height: 0; /* 确保flex子元素可以正确收缩 */
  width: 100%;
  box-sizing: border-box;
}

.chat-header {
  background: #fff;
  padding: 20px;
  text-align: center;
  border-bottom: 1px solid #e0e0e0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  flex-shrink: 0; /* 防止头部被压缩 */
}

.chat-header h1 {
  margin: 0 0 8px 0;
  color: #333;
  font-size: 24px;
}

.chat-header p {
  margin: 0 0 8px 0;
  color: #666;
  font-size: 14px;
}

.api-status {
  font-size: 12px;
  padding: 4px 12px;
  border-radius: 12px;
  background: #e8f5e8;
  color: #2d5a2d;
  display: inline-block;
}

.api-status.demo-mode {
  background: #fff3cd;
  color: #856404;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0; /* 确保可以正确滚动 */
  width: 100%;
  box-sizing: border-box;
}

.message {
  display: flex;
  margin-bottom: 16px;
}

.user-message {
  justify-content: flex-end;
}

.assistant-message {
  justify-content: flex-start;
}

.message-content {
  max-width: 70%;
  padding: 12px 16px;
  border-radius: 18px;
  position: relative;
  cursor: pointer;
  transition: all 0.3s ease;
}

.message-content:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.user-message .message-content {
  background: #007AFF;
  color: white;
  border-bottom-right-radius: 4px;
}

.assistant-message .message-content {
  background: white;
  color: #333;
  border: 1px solid #e0e0e0;
  border-bottom-left-radius: 4px;
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.message-preview {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
}

.role-indicator {
  font-size: 12px;
  font-weight: 600;
  opacity: 0.8;
  padding: 2px 6px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.2);
}

.assistant-message .role-indicator {
  background: rgba(0, 122, 255, 0.1);
  color: #007AFF;
}

.preview-text {
  font-size: 14px;
  opacity: 0.9;
  flex: 1;
  font-style: italic;
}

.expand-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  transition: transform 0.3s ease;
  opacity: 0.7;
}

.expand-icon.expanded {
  transform: rotate(180deg);
}

.message-content:hover .expand-icon {
  opacity: 1;
}

.message-text {
  margin-bottom: 8px;
  line-height: 1.4;
  word-wrap: break-word;
  font-size: 14px;
  opacity: 0;
  transform: translateY(-10px);
  transition: all 0.3s ease;
  padding: 8px 0;
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.assistant-message .message-text {
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

.message-text.fade-in {
  opacity: 1;
  transform: translateY(0);
}

.message-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
}

.message-time {
  font-size: 11px;
  opacity: 0.7;
}

.play-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 50%;
  background: rgba(0, 122, 255, 0.1);
  color: #007AFF;
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: 0.7;
}

.play-button:hover:not(:disabled) {
  background: rgba(0, 122, 255, 0.2);
  opacity: 1;
  transform: scale(1.1);
}

.play-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.play-button.playing {
  background: #FF3B30;
  color: white;
  animation: pulse 1s infinite;
}

.user-message .play-button {
  background: rgba(255, 255, 255, 0.2);
  color: white;
}

.user-message .play-button:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.3);
}

.input-container {
  background: white;
  padding: 20px;
  border-top: 1px solid #e0e0e0;
  flex-shrink: 0; /* 防止输入区域被压缩 */
  width: 100%;
  box-sizing: border-box;
}

.voice-input-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  width: 100%;
  box-sizing: border-box;
}

.status-text {
  font-size: 14px;
  color: #666;
  text-align: center;
  min-height: 20px;
}

.button-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
  align-items: center;
}

.main-buttons {
  display: flex;
  gap: 8px;
}

.record-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 32px;
  border: none;
  border-radius: 50px;
  background: #007AFF;
  color: white;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
}

.record-button:hover:not(:disabled) {
  background: #0056CC;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 122, 255, 0.4);
}

.record-button:disabled {
  background: #ccc;
  cursor: not-allowed;
  box-shadow: none;
}

.record-button.recording {
  background: #FF3B30;
  animation: pulse 1.5s infinite;
}

.stop-button {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 24px;
  border: none;
  border-radius: 50px;
  background: #FF3B30;
  color: white;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 12px rgba(255, 59, 48, 0.3);
}

.stop-button:hover {
  background: #D70015;
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(255, 59, 48, 0.4);
}

.global-reset {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border: none;
  border-radius: 50%;
  background: #6c757d;
  color: white;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(108, 117, 125, 0.3);
}

.global-reset:hover {
  background: #5a6268;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(108, 117, 125, 0.4);
}

.global-reset::before {
  content: "×";
  font-size: 24px;
  font-weight: bold;
  line-height: 1;
}

.record-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
}

.recording-animation {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: white;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
}

/* 统计面板样式 */
.stats-panel {
  background: white;
  border-top: 1px solid #e0e0e0;
  padding: 20px;
  flex-shrink: 0;
  width: 100%;
  box-sizing: border-box;
}

.stats-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.stats-header-controls {
  display: flex;
  gap: 8px; /* 按钮之间的间隔 */
  align-items: center;
}

.stats-header h3 {
  margin: 0;
  color: #333;
  font-size: 18px;
}

/* 折叠按钮样式 */
.stats-header .toggle-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 50%;
  background: rgba(0, 122, 255, 0.1);
  color: #007AFF;
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: 0.7;
}

.stats-header .toggle-button:hover {
  background: rgba(0, 122, 255, 0.2);
  opacity: 1;
  transform: scale(1.1);
}

/* 重置按钮样式调整 */
.stats-header .reset-button {
  padding: 6px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #f8f9fa;
  color: #666;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.stats-header .reset-button:hover {
  background: #e9ecef;
  border-color: #adb5bd;
}

/* 确保统计内容区域可以折叠 */
.stats-content {
  transition: all 0.3s ease;
}

.stats-content {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
  width: 100%;
  box-sizing: border-box;
}

.stat-item {
  text-align: center;
  padding: 12px;
  border-radius: 12px;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  box-sizing: border-box;
  min-height: 90px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.progress-item {
  grid-column: span 1;
}

.stat-label {
  font-size: 11px;
  color: #666;
  margin-bottom: 4px;
  font-weight: 500;
}

.stat-value {
  font-size: 16px;
  font-weight: 700;
  color: #007AFF;
  margin-bottom: 2px;
}

.stat-description {
  font-size: 10px;
  color: #999;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background: #e9ecef;
  border-radius: 2px;
  overflow: hidden;
  margin: 8px 0;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #007AFF, #0056CC);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 10px;
  color: #666;
  font-weight: 600;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .stats-content {
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  }
}

@media (max-width: 480px) {
  .stats-content {
    grid-template-columns: 1fr;
  }
  
  .stat-item {
    min-height: auto;
  }
}

.progress-item {
  grid-column: 1 / -1; /* 使进度条横跨整个宽度 */
  padding: 8px 12px; /* 减小内边距使高度更小 */
  min-height: auto; /* 移除最小高度限制 */
}

/* 调整进度条相关元素的样式 */
.progress-item .progress-bar {
  margin: 4px 0; /* 减小上下边距 */
}

.progress-item .stat-label,
.progress-item .progress-text {
  font-size: 10px; /* 统一使用小字号 */
}

.stat-label {
  font-size: 11px;
  color: #666;
  margin-bottom: 4px;
  font-weight: 500;
}

.stat-value {
  font-size: 16px;
  font-weight: 700;
  color: #007AFF;
  margin-bottom: 2px;
}

.stat-description {
  font-size: 10px;
  color: #999;
}

.progress-bar {
  width: 100%;
  height: 4px;
  background: #e9ecef;
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #007AFF, #0056CC);
  border-radius: 2px;
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 10px;
  color: #666;
  font-weight: 600;
}

/* 滚动条样式 */
.messages-container::-webkit-scrollbar {
  width: 6px;
}

.messages-container::-webkit-scrollbar-track {
  background: #f1f1f1;
}

.messages-container::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

.messages-container::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .chat-container {
    height: auto;
  }
  
  .message-content {
    max-width: 85%;
  }
  
  .record-button {
    padding: 14px 28px;
    font-size: 14px;
  }
  
  .message-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
  
  .expand-icon {
    align-self: flex-end;
  }
  
  .stats-content {
    flex-direction: column;
    gap: 10px;
  }
  
  .stat-item {
    min-width: auto;
    flex: none;
    width: 100%;
  }
  
  .progress-item {
    flex: none;
    min-width: auto;
    width: 100%;
  }
  
  .stats-panel {
    padding: 15px;
  }
  
  .stats-header {
    margin-bottom: 15px;
  }
  
  .settings-content {
    gap: 12px;
  }
  
  .setting-control {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }
  
  .setting-value {
    align-self: flex-end;
  }
  
  .voice-select {
    font-size: 11px;
    padding: 6px 10px;
  }
}

@media (max-width: 480px) {
  .stats-content {
    gap: 8px;
  }
  
  .stat-item {
    padding: 10px 12px;
  }
  
  .stat-value {
    font-size: 18px;
  }
  
  .stat-label {
    font-size: 11px;
  }
  
  .stat-description {
    font-size: 10px;
  }
  
  .settings-panel {
    padding: 15px;
  }
  
  .settings-header {
    margin-bottom: 15px;
  }
  
  .setting-label {
    font-size: 13px;
  }
  
  .setting-description {
    font-size: 9px;
  }
}

/* 设置面板样式 */
.settings-panel {
  background: white;
  border-top: 1px solid #e0e0e0;
  padding: 20px;
  flex-shrink: 0; /* 防止设置面板被压缩 */
  width: 100%;
  box-sizing: border-box;
}

.settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.settings-header h3 {
  margin: 0;
  color: #333;
  font-size: 18px;
}

.toggle-button {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 50%;
  background: rgba(0, 122, 255, 0.1);
  color: #007AFF;
  cursor: pointer;
  transition: all 0.2s ease;
  opacity: 0.7;
}

.toggle-button:hover:not(:disabled) {
  background: rgba(0, 122, 255, 0.2);
  opacity: 1;
  transform: scale(1.1);
}

.toggle-button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.settings-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  box-sizing: border-box;
}

.settings-row {
  display: flex;
  gap: 16px;
}

.setting-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  box-sizing: border-box;
  flex: 1;
}

.setting-label {
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

.setting-description {
  font-size: 10px;
  color: #999;
}

.setting-control {
  display: flex;
  align-items: center;
  gap: 8px;
}

.slider {
  width: 100%;
  height: 4px;
  background: #e9ecef;
  border-radius: 2px;
  overflow: hidden;
}

.setting-value {
  font-size: 12px;
  color: #666;
  font-weight: 500;
  min-width: 30px;
  text-align: right;
}

.toggle-switch {
  position: relative;
  width: 40px;
  height: 20px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.toggle-slider {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: #e9ecef;
  border-radius: 20px;
  transition: all 0.3s ease;
}

.toggle-slider::before {
  position: absolute;
  content: "";
  width: 16px;
  height: 16px;
  background: white;
  border-radius: 50%;
  transition: all 0.3s ease;
}

.toggle-switch input:checked + .toggle-slider {
  background: #007AFF;
}

.toggle-switch input:checked + .toggle-slider::before {
  transform: translateX(20px);
}

.voice-select {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  background: white;
  font-size: 12px;
  color: #333;
  cursor: pointer;
  transition: all 0.2s ease;
}

.voice-select:hover {
  border-color: #007AFF;
}

.voice-select:focus {
  outline: none;
  border-color: #007AFF;
  box-shadow: 0 0 0 2px rgba(0, 122, 255, 0.1);
}

.button-group {
  flex-direction: column;
  gap: 12px;
}
</style>
