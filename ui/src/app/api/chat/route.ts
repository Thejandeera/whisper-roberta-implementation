import {
  streamText,
  convertToModelMessages,
  createUIMessageStreamResponse,
  toUIMessageStream,
} from 'ai';
import { createOllama } from 'ai-sdk-ollama';

const ollama = createOllama({
  baseURL: 'http://localhost:11434',
});

const SYSTEM_PROMPT = `
You are Vanguard, an AI customer support agent for 'Zenvixor Studios'.
Your role is to assist users with questions about web development, video editing, and social media management services.

Rules:
1. Be concise, polite, and helpful. Keep responses under 3 sentences.
2. If the user expresses a negative emotion, immediately apologize and offer to escalate the issue to a human manager.
3. Do not invent pricing. If asked about prices, say they vary based on project scope and offer to schedule a consultation.
4. You are receiving transcribed voice text. It may contain errors. Infer the user's intent to the best of your ability.
`;

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: ollama('gemma3:4b') as any,
    system: SYSTEM_PROMPT,
    messages: await convertToModelMessages(messages),
  });

  return createUIMessageStreamResponse({
    stream: toUIMessageStream({ stream: result.stream }),
  });
}