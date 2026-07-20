"use client";

import { useState, useRef, useEffect } from "react";
import { useChat } from "@ai-sdk/react";
import "./model/model.css";

const EMOTION_COLORS: Record<string, { bg: string; text: string; glow: string }> = {
  admiration:     { bg: "rgba(251,191,36,0.15)",  text: "#fbbf24", glow: "0 0 24px rgba(251,191,36,0.3)" },
  amusement:      { bg: "rgba(52,211,153,0.15)",  text: "#34d399", glow: "0 0 24px rgba(52,211,153,0.3)" },
  anger:          { bg: "rgba(239,68,68,0.15)",   text: "#ef4444", glow: "0 0 24px rgba(239,68,68,0.3)" },
  annoyance:      { bg: "rgba(248,113,113,0.15)", text: "#f87171", glow: "0 0 24px rgba(248,113,113,0.3)" },
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
  return CATEGORY_COLORS[categoryName?.toLowerCase()] || CATEGORY_COLORS.neutral;
}

export default function ModelPage() {
  const [isRecording, setIsRecording] = useState(false);
  const [livePartialText, setLivePartialText] = useState("");
  const [error, setError] = useState<string | null>(null);
  
  // Track metadata (emotion tags) mapped to the specific text the user said
  const [emotionMeta, setEmotionMeta] = useState<Record<string, { emotion: string, sentiment: string }>>({});

  // Vercel AI SDK integration mapping to /api/chat
  const { messages, sendMessage, setMessages, status } = useChat();
  const isLoading = status === "streaming" || status === "submitted";

  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, livePartialText, isLoading]);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const startLiveStream = async () => {
    try {
      setError(null);
      setLivePartialText("");
      setMessages([]); // Clear chat history on new session
      setEmotionMeta({});

      wsRef.current = new WebSocket("ws://localhost:8001/live-stream");
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "partial") {
          setLivePartialText(data.text);
        } else if (data.type === "analyzed") {
          // 1. Save the emotion data linked to the exact text string
          setEmotionMeta(prev => ({
            ...prev,
            [data.text]: { emotion: data.emotion, sentiment: data.sentiment_category }
          }));

          // 2. Clear the partial listening text
          setLivePartialText(""); 

          // 3. Trigger Vercel AI SDK to hit Ollama automatically
          sendMessage({ text: data.text });
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
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
      console.error("Error accessing microphone:", err);
      setError("Microphone access denied or connection failed.");
    }
  };

  const stopLiveStream = () => {
    setIsRecording(false);
    if (processorRef.current && audioContextRef.current) {
      processorRef.current.disconnect();
      audioContextRef.current.close();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <main className="model-page">
      <div className="model-bg-blob model-bg-blob--1" />
      <div className="model-bg-blob model-bg-blob--2" />
      
      <div className="model-container" style={{ maxWidth: "800px" }}>
        <header className="model-header">
          <div className="model-header__badge">Vanguard AI · Voice Agent</div>
          <h1 className="model-header__title">Live Support Hub</h1>
          <p className="model-header__subtitle">
            Speak to interact with the Zenvixor Support AI.
          </p>
        </header>

        <section className="model-input-section" style={{ alignItems: "center", padding: "1.5rem" }}>
          {error && <div className="model-error"><span>⚠</span> {error}</div>}
          
          {!isRecording ? (
             <button className="model-btn-analyze" onClick={startLiveStream} style={{ width: "100%", padding: "1.25rem" }}>
                Start Voice Chat
             </button>
          ) : (
             <button className="model-btn" onClick={stopLiveStream} style={{ background: "rgba(239,68,68,0.2)", color: "#f87171", border: "1px solid rgba(239,68,68,0.4)", width: "100%", padding: "1.25rem" }}>
                End Call
             </button>
          )}
        </section>

        {(messages.length > 0 || livePartialText) && (
          <section className="model-result-section" style={{ display: "flex", flexDirection: "column", height: "500px" }}>
            <div style={{ flex: 1, overflowY: "auto", paddingRight: "10px", display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              
              {messages.map((m) => {
                // Extract text content from parts
                const textContent = m.parts
                  .filter((part) => part.type === "text")
                  .map((part) => part.text)
                  .join("");

                // Determine styling based on role
                if (m.role === 'user') {
                  const meta = emotionMeta[textContent];
                  const emoStyle = getEmotionStyle(meta?.emotion || "neutral");
                  const catStyle = getCategoryStyle(meta?.sentiment || "neutral");
                  
                  return (
                    <div key={m.id} style={{ alignSelf: "flex-end", maxWidth: "80%" }}>
                       <div style={{ fontSize: "0.75rem", color: "#9ca3af", textAlign: "right", marginBottom: "4px" }}>
                          You 
                          {meta && (
                            <>
                              <span style={{ marginLeft: "8px", background: catStyle.bg, color: catStyle.text, padding: "2px 6px", borderRadius: "4px", fontSize: "0.65rem" }}>{meta.sentiment.toUpperCase()}</span>
                              <span style={{ marginLeft: "4px", background: emoStyle.bg, color: emoStyle.text, padding: "2px 6px", borderRadius: "4px", fontSize: "0.65rem" }}>{meta.emotion.toUpperCase()}</span>
                            </>
                          )}
                       </div>
                       <div style={{ background: "#4f46e5", color: "white", padding: "1rem", borderRadius: "12px 12px 0 12px", lineHeight: "1.5" }}>
                         "{textContent}"
                       </div>
                    </div>
                  );
                }

                // AI Assistant Message (Streaming)
                return (
                  <div key={m.id} style={{ alignSelf: "flex-start", maxWidth: "80%" }}>
                     <div style={{ fontSize: "0.75rem", color: "#9ca3af", marginBottom: "4px" }}>Vanguard AI</div>
                     <div style={{ background: "rgba(15,23,42,0.8)", border: "1px solid rgba(255,255,255,0.1)", color: "#e2e8f0", padding: "1rem", borderRadius: "12px 12px 12px 0", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                       {textContent}
                     </div>
                  </div>
                );
              })}

              {/* Live Listening State */}
              {livePartialText && (
                 <div style={{ alignSelf: "flex-end", maxWidth: "80%", opacity: 0.7 }}>
                   <div style={{ background: "rgba(79,70,229,0.3)", color: "#a5b4fc", padding: "0.75rem 1rem", borderRadius: "12px 12px 0 12px", fontStyle: "italic" }}>
                     "{livePartialText}..."
                   </div>
                 </div>
              )}

              {/* Agent Thinking State (Triggered when hook is loading and last message is from user) */}
              {isLoading && messages[messages.length - 1]?.role === 'user' && (
                 <div style={{ alignSelf: "flex-start", maxWidth: "80%" }}>
                   <div style={{ background: "rgba(15,23,42,0.5)", border: "1px solid rgba(255,255,255,0.05)", color: "#94a3b8", padding: "0.75rem 1rem", borderRadius: "12px 12px 12px 0", display: "flex", gap: "8px", alignItems: "center" }}>
                     <span className="model-live-dot"></span> Vanguard is generating a response...
                   </div>
                 </div>
              )}

              <div ref={chatEndRef} />
            </div>
          </section>
        )}
      </div>
    </main>
  );
}