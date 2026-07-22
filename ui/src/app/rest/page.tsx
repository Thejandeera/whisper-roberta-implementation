"use client";

import { useState, useRef, ChangeEvent } from "react";
import "./model.css";

export default function RestApiPage() {
  const [file, setFile] = useState<File | null>(null);
  const [base64String, setBase64String] = useState<string>("");
  const [transcript, setTranscript] = useState<string>("");
  const [processingTime, setProcessingTime] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [copyText, setCopyText] = useState<string>("Copy Base64");

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setTranscript("");
    setProcessingTime(null);
    setError(null);
    setCopyText("Copy Base64");

    const reader = new FileReader();
    reader.onload = (event) => {
      const result = event.target?.result as string;
      const b64 = result.split(",")[1];
      setBase64String(b64);
    };
    reader.onerror = () => {
      setError("Failed to read file.");
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleSend = async () => {
    if (!base64String) {
      setError("Please select an audio file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setTranscript("");
    setProcessingTime(null);

    try {
      const response = await fetch("http://localhost:8001/api/v1/analyze-audio", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ audio_data: base64String }),
      });

      const data = await response.json();

      if (data.status === "success") {
        setTranscript(data.transcript);
        setProcessingTime(data.processing_time_seconds);
      } else {
        setError(data.error || "An error occurred during transcription.");
      }
    } catch (err) {
      setError("Failed to connect to the backend server.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!base64String) return;
    try {
      await navigator.clipboard.writeText(base64String);
      setCopyText("Copied!");
      setTimeout(() => setCopyText("Copy Base64"), 2000);
    } catch (err) {
      setError("Failed to copy text.");
    }
  };

  return (
    <main className="rest-page">
      <div className="rest-container">
        <h1 className="rest-title">Whisper REST API Tester</h1>

        <section className="rest-section">
          <h2 className="rest-section-title">1. Select Audio File</h2>
          <div className="rest-input-group">
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="rest-file-input"
            />
            <button
              onClick={handleSend}
              disabled={isLoading || !base64String}
              className="rest-button"
            >
              {isLoading ? "Processing..." : "Send to Backend"}
            </button>
          </div>
          {error && <p className="rest-error">{error}</p>}
        </section>

        <section className="rest-section">
          <h2 className="rest-section-title">2. Results</h2>
          <div className="rest-results-grid">
            <div className="rest-box">
              <h3>Processing Time</h3>
              <p className="rest-stat">
                {processingTime !== null ? `${processingTime} seconds` : "--"}
              </p>
            </div>
            <div className="rest-box rest-transcript-box">
              <h3>Transcript</h3>
              <p className="rest-transcript-text">
                {transcript || "No transcript generated yet."}
              </p>
            </div>
          </div>
        </section>

        <section className="rest-section">
          <div className="rest-base64-header">
            <h2 className="rest-section-title">3. Base64 Payload</h2>
            <button
              onClick={handleCopy}
              disabled={!base64String}
              className="rest-button rest-copy-button"
            >
              {copyText}
            </button>
          </div>
          <div className="rest-base64-container">
            {base64String ? (
              <p className="rest-base64-text">{base64String}</p>
            ) : (
              <p className="rest-placeholder">Select a file to generate Base64 string...</p>
            )}
          </div>
        </section>
      </div>
    </main>
  );
}