"use client";

import { useState, useRef, useEffect } from "react";
import { useChat } from "@ai-sdk/react";
import "../model/model.css"; 

interface AnalysisResult {
  text: string;
  emotion: string;
  sentiment_category: string;
  confidence: number;
}

const EMOTION_COLORS: Record<string, { bg: string; text: string; glow: string }> = {
  admiration:     { bg: "rgba(251,191,36,0.15)",  text: "#fbbf24", glow: "0 0 24px rgba(251,191,36,0.3)" },
  amusement:      { bg: "rgba(52,211,153,0.15)",  text: "#34d399", glow: "0 0 24px rgba(52,211,153,0.3)" },
  anger:          { bg: "rgba(239,68,68,0.15)",   text: "#ef4444", glow: "0 0 24px rgba(239,68,68,0.3)" },
  annoyance:      { bg: "rgba(248,113,113,0.15)", text: "#f87171", glow: "0 0 24px rgba(248,113,113,0.3)" },
  approval:       { bg: "rgba(16,185,129,0.15)",  text: "#10b981", glow: "0 0 24px rgba(16,185,129,0.3)" },
  caring:         { bg: "rgba(244,114,182,0.15)", text: "#f472b6", glow: "0 0 24px rgba(244,114,182,0.3)" },
  confusion:      { bg: "rgba(156,163,175,0.15)", text: "#9ca3af", glow: "0 0 24px rgba(156,163,175,0.3)" },
  curiosity:      { bg: "rgba(96,165,250,0.15)",  text: "#60a5fa", glow: "0 0 24px rgba(96,165,250,0.3)" },
  desire:         { bg: "rgba(236,72,153,0.15)",  text: "#ec4899", glow: "0 0 24px rgba(236,72,153,0.3)" },
  disappointment: { bg: "rgba(167,139,250,0.15)", text: "#a78bfa", glow: "0 0 24px rgba(167,139,250,0.3)" },
  disapproval:    { bg: "rgba(251,146,60,0.15)",  text: "#fb923c", glow: "0 0 24px rgba(251,146,60,0.3)" },
  disgust:        { bg: "rgba(132,204,22,0.15)",  text: "#84cc16", glow: "0 0 24px rgba(132,204,22,0.3)" },
  embarrassment:  { bg: "rgba(251,113,133,0.15)", text: "#fb7185", glow: "0 0 24px rgba(251,113,133,0.3)" },
  excitement:     { bg: "rgba(250,204,21,0.15)",  text: "#facc15", glow: "0 0 24px rgba(250,204,21,0.3)" },
  fear:           { bg: "rgba(75,85,99,0.15)",    text: "#4b5563", glow: "0 0 24px rgba(75,85,99,0.3)" },
  gratitude:      { bg: "rgba(5,150,105,0.15)",   text: "#059669", glow: "0 0 24px rgba(5,150,105,0.3)" },
  grief:          { bg: "rgba(31,41,55,0.2)",     text: "#6b7280", glow: "0 0 24px rgba(31,41,55,0.3)" },
  joy:            { bg: "rgba(253,224,71,0.15)",  text: "#fde047", glow: "0 0 24px rgba(253,224,71,0.3)" },
  love:           { bg: "rgba(225,29,72,0.15)",   text: "#e11d48", glow: "0 0 24px rgba(225,29,72,0.3)" },
  nervousness:    { bg: "rgba(217,119,6,0.15)",   text: "#d97706", glow: "0 0 24px rgba(217,119,6,0.3)" },
  optimism:       { bg: "rgba(56,189,248,0.15)",  text: "#38bdf8", glow: "0 0 24px rgba(56,189,248,0.3)" },
  pride:          { bg: "rgba(129,140,248,0.15)", text: "#818cf8", glow: "0 0 24px rgba(129,140,248,0.3)" },
  realization:    { bg: "rgba(14,165,233,0.15)",  text: "#0ea5e9", glow: "0 0 24px rgba(14,165,233,0.3)" },
  relief:         { bg: "rgba(20,184,166,0.15)",  text: "#14b8a6", glow: "0 0 24px rgba(20,184,166,0.3)" },
  remorse:        { bg: "rgba(113,113,122,0.15)", text: "#71717a", glow: "0 0 24px rgba(113,113,122,0.3)" },
  sadness:        { bg: "rgba(59,130,246,0.15)",  text: "#3b82f6", glow: "0 0 24px rgba(59,130,246,0.3)" },
  surprise:       { bg: "rgba(192,132,252,0.15)", text: "#c084fc", glow: "0 0 24px rgba(192,132,252,0.3)" },
  neutral:        { bg: "rgba(156,163,175,0.1)",  text: "#9ca3af", glow: "none" }
};

const CATEGORY_COLORS: Record<string, { bg: string; text: string }> = {
  positive: { bg: "rgba(16, 185, 129, 0.2)", text: "#10b981" },
  negative: { bg: "rgba(239, 68, 68, 0.2)", text: "#ef4444" },
  neutral:  { bg: "rgba(156, 163, 175, 0.2)", text: "#9ca3af" }
};

const DEFAULT_EMOTION_STYLE = { bg: "rgba(99,102,241,0.15)", text: "#818cf8", glow: "0 0 24px rgba(99,102,241,0.3)" };

function getEmotionStyle(emotion: string) {
  return EMOTION_COLORS[emotion?.toLowerCase()] ?? DEFAULT_EMOTION_STYLE;
}

function getCategoryStyle(categoryName: string) {
  const normalized = categoryName?.toLowerCase();
  return CATEGORY_COLORS[normalized] || CATEGORY_COLORS.neutral;
}

export default function LiveModelPage() {
  const [isRecording, setIsRecording] = useState(false);
  const isRecordingRef = useRef(false);

  const [livePartialText, setLivePartialText] = useState("");
  const [analyzedSentences, setAnalyzedSentences] = useState<AnalysisResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const { messages, setMessages } = useChat();
  
  const [emotionMeta, setEmotionMeta] = useState<Record<string, { emotion: string, sentiment: string }>>({});

  const [accumulatedText, setAccumulatedText] = useState("");
  const accumulatedTextRef = useRef("");

  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const tableRef = useRef<HTMLDivElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const silenceTimerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (tableRef.current) {
      tableRef.current.scrollTop = tableRef.current.scrollHeight;
    }
  }, [analyzedSentences]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, livePartialText, accumulatedText]);

  useEffect(() => {
    return () => {
      isRecordingRef.current = false;
      if (wsRef.current) wsRef.current.close();
      if (silenceTimerRef.current) clearTimeout(silenceTimerRef.current);
    };
  }, []);

  const submitAccumulatedSpeech = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }

    const finalSpeech = accumulatedTextRef.current.trim();
    if (finalSpeech) {
      try {
        setMessages((prev: any) => [
          ...prev,
          { id: Date.now().toString(), role: "user", content: finalSpeech }
        ]);
      } catch (err) {}
    }
    
    accumulatedTextRef.current = "";
    setAccumulatedText("");
    setLivePartialText("");
  };

  const resetSilenceTimer = () => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
    }
    
    silenceTimerRef.current = setTimeout(() => {
      submitAccumulatedSpeech();
    }, 3000);
  };

  const startLiveStream = async () => {
    try {
      setError(null);
      setLivePartialText("");
      setAccumulatedText("");
      accumulatedTextRef.current = "";
      setAnalyzedSentences([]);
      setMessages([]); 
      setEmotionMeta({});
      
      isRecordingRef.current = true;

      if (silenceTimerRef.current) {
        clearTimeout(silenceTimerRef.current);
        silenceTimerRef.current = null;
      }

      wsRef.current = new WebSocket("ws://localhost:8001/live-stream-vad");
      
      wsRef.current.onmessage = (event) => {
        if (!isRecordingRef.current) return;

        const data = JSON.parse(event.data);
        const cleanText = data.text ? data.text.trim() : "";

        if (cleanText) {
          resetSilenceTimer();
        }

        if (data.type === "partial") {
          setLivePartialText(data.text);
        } else if (data.type === "analyzed") {
          setAnalyzedSentences((prev) => [...prev, data]);
          
          setEmotionMeta(prev => ({
            ...prev,
            [data.text]: { emotion: data.emotion, sentiment: data.sentiment_category }
          }));

          setLivePartialText(""); 

          if (cleanText) {
            const separator = accumulatedTextRef.current ? " " : "";
            accumulatedTextRef.current += separator + cleanText;
            setAccumulatedText(accumulatedTextRef.current);
          }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(512, 1, 1);
      processorRef.current = processor;

      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          const float32Array = e.inputBuffer.getChannelData(0);
          const int16Array = new Int16Array(float32Array.length);
          for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          wsRef.current.send(int16Array.buffer);
        }
      };

      setIsRecording(true);

    } catch (err: any) {
      setError("Microphone access denied or connection failed.");
      isRecordingRef.current = false;
      setIsRecording(false);
    }
  };

  const stopLiveStream = () => {
    isRecordingRef.current = false;
    setIsRecording(false);
    
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    
    try {
      if (processorRef.current) {
        processorRef.current.disconnect();
      }
    } catch (e) {}
    
    try {
      if (audioContextRef.current && audioContextRef.current.state !== "closed") {
        audioContextRef.current.close();
      }
    } catch (e) {}
    
    try {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          try {
            track.stop();
          } catch (err) {}
        });
      }
    } catch (e) {}

    try {
      if (wsRef.current) {
        wsRef.current.close();
      }
    } catch (e) {}

    submitAccumulatedSpeech();
  };

  return (
    <main className="model-page">
      <div className="model-bg-blob model-bg-blob--1" />
      <div className="model-bg-blob model-bg-blob--2" />
      
      <div className="model-container">
        <header className="model-header">
          <div className="model-header__badge">Vanguard AI · Live Voice Hub</div>
          <h1 className="model-header__title">Support Voice Agent</h1>
          <p className="model-header__subtitle">
            Speak to interact with Zenvixor Studios' AI customer agent and see real-time emotional analysis.
          </p>
        </header>

        <section className="model-input-section" style={{ alignItems: "center", padding: "1.5rem" }}>
          {error && <div className="model-error"><span>⚠</span> {error}</div>}
          
          {!isRecording ? (
             <button className="model-btn-analyze" onClick={startLiveStream} style={{ width: "100%", padding: "1.25rem" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: "8px", verticalAlign: "middle" }}>
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                  <line x1="12" y1="19" x2="12" y2="22"/>
                </svg>
                Start Voice Chat
             </button>
          ) : (
             <button className="model-btn" onClick={stopLiveStream} style={{ background: "rgba(239,68,68,0.2)", color: "#f87171", border: "1px solid rgba(239,68,68,0.4)", width: "100%", padding: "1.25rem" }}>
                <div className="model-recording-pulse" style={{ marginRight: "8px", display: "inline-block" }}></div>
                End Call & Send Message
             </button>
          )}
        </section>

        {(messages.length > 0 || livePartialText || accumulatedText || analyzedSentences.length > 0) && (
          <div className="model-grid">
            
            <div className="model-chat-container">
              <h3 style={{ fontSize: "1.2rem", fontWeight: "700", color: "#ffffff", marginBottom: "1rem" }}>Support Agent Chat</h3>
              
              <div className="model-chat-log">
                {messages.map((m: any) => {
                  const textContent = m.parts
                    ? m.parts
                        .filter((part: any) => part.type === "text")
                        .map((part: any) => part.text)
                        .join("")
                    : m.content;
                    
                  if (m.role === 'user') {
                    const userSentences = textContent.split(/(?<=[.!?])\s+/);
                    
                    return (
                      <div key={m.id} className="model-chat-bubble model-chat-bubble--user">
                        <div className="model-chat-bubble-meta model-chat-bubble-meta--user">
                          <span>You</span>
                        </div>
                        <div className="model-chat-text model-chat-text--user">
                          {userSentences.map((sentence: string, sIdx: number) => {
                            const trimmed = sentence.trim();
                            if (!trimmed) return null;
                            const meta = emotionMeta[trimmed] || emotionMeta[trimmed + "."] || emotionMeta[trimmed.replace(/\.$/, "")];
                            const emoStyle = meta ? getEmotionStyle(meta.emotion) : null;
                            const catStyle = meta ? getCategoryStyle(meta.sentiment) : null;
                            
                            return (
                              <span key={sIdx} style={{ display: "block", marginBottom: sIdx < userSentences.length - 1 ? "8px" : "0" }}>
                                "{trimmed}"
                                {meta && (
                                  <span style={{ display: "inline-flex", gap: "4px", marginLeft: "8px", verticalAlign: "middle" }}>
                                    <span className="model-table-badge" style={{ background: catStyle!.bg, color: catStyle!.text, border: `1px solid ${catStyle!.text}40`, padding: "1px 5px", fontSize: "0.55rem", lineHeight: "1", textTransform: "uppercase" }}>
                                      {meta.sentiment.toUpperCase()}
                                    </span>
                                    <span className="model-table-badge" style={{ background: emoStyle!.bg, color: emoStyle!.text, padding: "1px 5px", fontSize: "0.55rem", lineHeight: "1", textTransform: "uppercase" }}>
                                      {meta.emotion.toUpperCase()}
                                    </span>
                                  </span>
                                )}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    );
                  }
                  
                  return (
                    <div key={m.id} className="model-chat-bubble model-chat-bubble--ai">
                      <div className="model-chat-bubble-meta">
                        <span>Vanguard AI</span>
                      </div>
                      <div className="model-chat-text model-chat-text--ai">
                        {textContent}
                      </div>
                    </div>
                  );
                })}

                {(accumulatedText || livePartialText) && isRecording && (
                  <div className="model-chat-bubble model-chat-bubble--user" style={{ opacity: 0.85 }}>
                    <div className="model-chat-bubble-meta model-chat-bubble-meta--user">
                      <span>You (Speaking...)</span>
                    </div>
                    <div className="model-chat-text model-chat-text--user" style={{ borderStyle: "dashed", borderColor: "rgba(255,255,255,0.3)" }}>
                      "{accumulatedText}{livePartialText ? " " + livePartialText : ""}..."
                    </div>
                  </div>
                )}
                
                <div ref={chatEndRef} />
              </div>
            </div>

            <section className="model-result-section" style={{ height: "500px", display: "flex", flexDirection: "column", padding: "1.5rem" }}>
              <h3 style={{ fontSize: "1.2rem", fontWeight: "700", color: "#ffffff", marginBottom: "1rem" }}>Inference Log Table</h3>
              
              <div className="model-table-container" ref={tableRef} style={{ flex: 1 }}>
                <table className="model-table">
                  <thead>
                    <tr>
                      <th style={{ width: "55%" }}>Transcribed Text</th>
                      <th style={{ width: "22%" }}>Sentiment</th>
                      <th style={{ width: "23%" }}>Emotion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyzedSentences.map((block, idx) => {
                      const emoStyle = getEmotionStyle(block.emotion);
                      const catStyle = getCategoryStyle(block.sentiment_category);
                      
                      return (
                        <tr key={idx}>
                          <td className="model-table-text" style={{ fontSize: "0.85rem" }}>"{block.text}"</td>
                          <td>
                            <span className="model-table-badge" style={{ background: catStyle.bg, color: catStyle.text, border: `1px solid ${catStyle.text}40`, fontSize: "0.65rem" }}>
                              {block.sentiment_category.toUpperCase()}
                            </span>
                          </td>
                          <td>
                            <span className="model-table-badge" style={{ background: emoStyle.bg, color: emoStyle.text, fontSize: "0.65rem" }}>
                              {block.emotion}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
      </div>
    </main>
  );
}