"use client";

import { useMemo, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import { linkCitationsHtml, type Citation } from "@/lib/citations";
import { renderMarkdown } from "@/lib/markdown";

const STATUS_STEPS = [
  { step: 1, text: "Teen andmebaasi päringut" },
  { step: 2, text: "Loen vastuseid" },
  { step: 3, text: "Loon vastust" }
];

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  scope?: string;
};

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [currentStep, setCurrentStep] = useState(0);
  const [statusOpen, setStatusOpen] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const assistantCounter = useRef(0);
  const activeAssistantId = useRef<string | null>(null);

  const hasMessages = messages.length > 0;

  const handleSubmit = async (text: string) => {
    const query = text.trim();
    if (!query || isStreaming) {
      return;
    }

    if (window.location.hash) {
      history.replaceState(null, "", window.location.pathname);
    }

    setIsStreaming(true);
    setCurrentStep(0);
    setStatusOpen(true);

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: query
    };
    const assistantId = `assistant-${Date.now()}-${assistantCounter.current++}`;
    const scope = `s${Date.now()}-${assistantCounter.current}`;

    const assistantMessage: Message = {
      id: assistantId,
      role: "assistant",
      content: "",
      citations: [],
      scope
    };

    const historyMessages = [...messages, userMessage].map((message) => ({
      role: message.role,
      content: message.content
    }));

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    activeAssistantId.current = assistantId;

    let answerStarted = false;

    await streamChat(
      { query, messages: historyMessages },
      (event) => {
        if (activeAssistantId.current !== assistantId) {
          return;
        }
        if (event.event === "status") {
          const step = Number(event.data.step || 0);
          if (step > 0) {
            setCurrentStep(step);
            setStatusOpen(true);
          }
          return;
        }
        if (event.event === "answer") {
          const delta = String(event.data.delta || "");
          if (!delta) {
            return;
          }
          if (!answerStarted) {
            setStatusOpen(false);
            answerStarted = true;
          }
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? { ...message, content: message.content + delta }
                : message
            )
          );
          return;
        }
        if (event.event === "citations") {
          const citations = (event.data.citations as Citation[]) || [];
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId ? { ...message, citations } : message
            )
          );
          return;
        }
        if (event.event === "final") {
          const answer = String(event.data.answer || "");
          const citations = (event.data.citations as Citation[]) || [];
          setStatusOpen(false);
          setMessages((prev) =>
            prev.map((message) =>
              message.id === assistantId
                ? { ...message, content: answer, citations }
                : message
            )
          );
          return;
        }
        if (event.event === "done") {
          setIsStreaming(false);
          setStatusOpen(false);
        }
      },
      () => {
        setIsStreaming(false);
        setStatusOpen(false);
      }
    );
  };

  const handleNewChat = () => {
    setMessages([]);
    setInput("");
    setCurrentStep(0);
    setStatusOpen(false);
    setIsStreaming(false);
    activeAssistantId.current = null;
    if (window.location.hash) {
      history.replaceState(null, "", window.location.pathname);
    }
  };

  const renderedMessages = useMemo(
    () =>
      messages.map((message) => {
        if (message.role === "assistant") {
          const html = linkCitationsHtml(
            renderMarkdown(message.content),
            message.citations || [],
            message.scope || ""
          );
          return { ...message, rendered: html };
        }
        return { ...message, rendered: renderMarkdown(message.content) };
      }),
    [messages]
  );

  return (
    <main>
      <div
        className="app-shell"
        data-has-messages={hasMessages || isStreaming}
      >
        <header className="topbar">
          <div className="topbar-brand">
            <h1 className="topbar-title">Kliiniline ravijuhiste assistent</h1>
            <p className="topbar-subtitle">
              Juhendipõhised vastused koos viidetega.
            </p>
          </div>
          <button className="btn btn-primary" onClick={handleNewChat}>
            Uus vestlus
          </button>
        </header>

        {statusOpen && (
          <section className="status-box">
            <div className="status-title">Loon vastust</div>
            <ul className="status-steps">
              {STATUS_STEPS.map((step) => (
                <li
                  key={step.step}
                  className={`status-step ${
                    currentStep >= step.step ? "done" : ""
                  }`}
                >
                  <span className="status-dot" />
                  <span>{`(${step.step}) ${step.text}`}</span>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="chat">
          {renderedMessages.map((message) => (
            <div key={message.id} className={`message ${message.role}`}>
              <div
                className="message-bubble"
                dangerouslySetInnerHTML={{ __html: message.rendered }}
              />
              {message.role === "assistant" && message.citations?.length ? (
                <div className="citations">
                  <h3>Viited</h3>
                  {message.citations.map((citation) => (
                    <div
                      key={`${message.scope}-${citation.display_index}`}
                      id={`citation-${message.scope}-${citation.display_index}`}
                      className="citation-card"
                    >
                      <a
                        className="citation-title"
                        href={`#citation-${message.scope}-${citation.display_index}`}
                      >
                        <span className="citation-index">
                          [{citation.display_index}]
                        </span>
                        <span>{citation.label}</span>
                      </a>
                      <div
                        className="citation-body"
                        dangerouslySetInnerHTML={{
                          __html: renderMarkdown(citation.text)
                        }}
                      />
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
        </section>

        <div className="input-shell">
          <form
            className="input-panel"
            onSubmit={(event) => {
              event.preventDefault();
              handleSubmit(input);
            }}
          >
            <textarea
              rows={1}
              placeholder="Küsi meditsiinialane küsimus..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSubmit(input);
                }
              }}
            />
            <button type="submit" disabled={!input.trim() || isStreaming}>
              Saada
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
