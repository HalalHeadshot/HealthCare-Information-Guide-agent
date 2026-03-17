import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endOfMessagesRef = useRef(null);
  
  // A consistent session ID per reload
  const [sessionId] = useState(() => Math.random().toString(36).substring(7));

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const resp = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.text, session_id: sessionId })
      });
      
      const data = await resp.json();
      const agentMsg = {
        sender: 'agent',
        text: data.output,
        steps: data.intermediate_steps
      };
      setMessages(prev => [...prev, agentMsg]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'agent', text: 'Error connecting to the agent backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-3xl flex-1 flex flex-col bg-surface rounded-xl shadow-2xl overflow-hidden border border-gray-800">
        
        {/* Header */}
        <div className="bg-gray-900 border-b border-gray-800 p-4 shadow-[0_4px_10px_rgba(0,243,255,0.1)]">
          <h1 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-neon-cyan to-neon-pink">
            Healthcare Information Guide Agent
          </h1>
          <p className="text-xs text-gray-400 mt-1">
            ⚠ Not a diagnostic tool. Consult a qualified professional for medical advice.
          </p>
        </div>

        {/* Chat History */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
             <div className="h-full flex items-center justify-center text-gray-500">
               <p>Send a message detailing your symptoms to get started.</p>
             </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div 
                className={`max-w-[85%] p-3 rounded-lg ${
                  msg.sender === 'user' 
                  ? 'bg-gray-800 text-neon-cyan/90 border border-neon-cyan/30 shadow-[0_0_10px_rgba(0,243,255,0.1)]' 
                  : 'bg-gray-900 text-gray-300 border border-gray-700'
                }`}
              >
                <div className="whitespace-pre-wrap leading-relaxed">{msg.text}</div>
                
                {/* Reasoning Trace (Agent) */}
                {msg.steps && msg.steps.length > 0 && (
                  <details className="mt-4 pt-3 border-t border-gray-700 text-xs">
                    <summary className="cursor-pointer text-gray-500 hover:text-gray-300 select-none">
                      Show reasoning trace ({msg.steps.length} steps)
                    </summary>
                    <div className="mt-2 space-y-2 bg-black/50 p-2 rounded">
                      {msg.steps.map((step, si) => (
                        <div key={si} className="border-l-2 border-neon-pink/50 pl-2">
                          <p className="text-neon-pink/80 font-semibold">{step.tool}</p>
                          <p className="text-gray-400">Input: {step.tool_input}</p>
                          <p className="text-gray-500 mt-1 line-clamp-2 hover:line-clamp-none transition-all">
                            Obs: {step.observation}
                          </p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="p-3 bg-gray-900 rounded-lg border border-neon-cyan/20 flex gap-2 items-center">
                <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse"></div>
                <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse delay-75"></div>
                <div className="w-2 h-2 rounded-full bg-neon-cyan animate-pulse delay-150"></div>
              </div>
            </div>
          )}
          <div ref={endOfMessagesRef} />
        </div>

        {/* Input Form */}
        <form onSubmit={sendMessage} className="bg-gray-900 p-4 border-t border-gray-800 flex gap-3">
          <input
            type="text"
            className="flex-1 bg-black border border-gray-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-neon-cyan focus:ring-1 focus:ring-neon-cyan shadow-[inset_0_2px_4px_rgba(0,0,0,0.6)] placeholder-gray-600 transition-all"
            placeholder="E.g., I have a fever and headache..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-gray-800 text-neon-cyan border border-neon-cyan/50 rounded-lg font-medium hover:bg-neon-cyan/10 hover:shadow-neon-cyan hover:-translate-y-0.5 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  )
}

export default App;
