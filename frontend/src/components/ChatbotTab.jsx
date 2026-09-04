import React, { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Trash2, Sparkles, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ChatbotTab({ clauses = [], onSendMessage, isThinking, chatHistory, onClearChat }) {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  const suggestedQuestions = [
    'What are the high-risk clauses?',
    'Is there a termination penalty?',
    'Summarize the key risks.',
    'What should I negotiate first?',
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isThinking]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isThinking) return;
    const q = inputValue.trim();
    setInputValue('');
    onSendMessage(q);
  };

  const handleSuggestedClick = (q) => {
    if (isThinking) return;
    onSendMessage(q);
  };

  return (
    <div className="bg-white rounded-2xl border border-stone-200/90 shadow-sm overflow-hidden flex flex-col h-[650px]">
      
      {/* Chat Header */}
      <div className="px-6 py-4 border-b border-stone-100 flex items-center justify-between bg-stone-50/50">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-[#1b4332] text-white flex items-center justify-center">
            <Bot className="w-4 h-4 text-emerald-300" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-stone-900 flex items-center space-x-2">
              <span>AI Contract Assistant</span>
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            </h3>
            <p className="text-[11px] text-stone-400 font-mono">
              Powered by Groq LLM (llama-3.1-8b-instant)
            </p>
          </div>
        </div>

        {chatHistory.length > 0 && (
          <button
            type="button"
            onClick={onClearChat}
            className="inline-flex items-center space-x-1 text-xs text-stone-400 hover:text-rose-600 transition-colors px-2 py-1 rounded-md"
            title="Clear Chat History"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* Message Stream */}
      <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-stone-50/30">
        {chatHistory.length === 0 ? (
          <div className="max-w-xl mx-auto py-8 text-center space-y-4">
            <div className="w-12 h-12 rounded-2xl bg-emerald-100/70 text-emerald-800 flex items-center justify-center mx-auto shadow-xs">
              <Sparkles className="w-6 h-6 text-emerald-700" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-stone-900">
                Hi! I've analyzed your contract.
              </h4>
              <p className="text-xs text-stone-500 mt-1 max-w-md mx-auto leading-relaxed">
                Ask me anything about specific clauses, liability exposures, or negotiation strategies.
              </p>
            </div>

            {/* Suggested Chips */}
            <div className="pt-2 flex flex-wrap gap-2 justify-center max-w-md mx-auto">
              {suggestedQuestions.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSuggestedClick(q)}
                  className="text-xs bg-white hover:bg-stone-100 text-stone-700 font-medium px-3 py-1.5 rounded-full border border-stone-200/90 shadow-2xs transition-all active:scale-95"
                >
                  "{q}"
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {chatHistory.map((msg, index) => {
              const isUser = msg.role === 'user';
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`flex items-start space-x-3 ${isUser ? 'justify-end' : 'justify-start'}`}
                >
                  {!isUser && (
                    <div className="w-7 h-7 rounded-lg bg-[#1b4332] text-white flex items-center justify-center shrink-0 mt-0.5">
                      <Bot className="w-3.5 h-3.5 text-emerald-300" />
                    </div>
                  )}

                  <div
                    className={`max-w-lg p-3.5 rounded-2xl text-xs leading-relaxed ${
                      isUser
                        ? 'bg-[#1b4332] text-white rounded-tr-xs shadow-xs'
                        : 'bg-white border border-stone-200 text-stone-800 rounded-tl-xs shadow-xs'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>

                  {isUser && (
                    <div className="w-7 h-7 rounded-lg bg-stone-200 text-stone-700 flex items-center justify-center shrink-0 mt-0.5 font-mono text-xs font-bold">
                      <User className="w-3.5 h-3.5" />
                    </div>
                  )}
                </motion.div>
              );
            })}

            {/* Thinking indicator */}
            {isThinking && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start space-x-3 justify-start"
              >
                <div className="w-7 h-7 rounded-lg bg-[#1b4332] text-white flex items-center justify-center shrink-0 mt-0.5">
                  <Bot className="w-3.5 h-3.5 text-emerald-300" />
                </div>
                <div className="bg-white border border-stone-200 p-3 rounded-2xl rounded-tl-xs shadow-xs flex items-center space-x-2 text-xs text-stone-500">
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-emerald-700" />
                  <span>Reviewing analyzed clauses...</span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-stone-100 bg-white flex items-center space-x-3">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Ask a question about the contract clauses or risks..."
          disabled={isThinking}
          className="flex-1 text-xs py-2.5 px-4 rounded-xl border border-stone-200 focus:border-emerald-700 focus:ring-1 focus:ring-emerald-700 outline-none transition-all placeholder:text-stone-400 bg-stone-50/50"
        />
        <button
          type="submit"
          disabled={!inputValue.trim() || isThinking}
          className={`p-2.5 rounded-xl font-semibold text-xs tracking-wide transition-all flex items-center justify-center ${
            inputValue.trim() && !isThinking
              ? 'bg-[#1b4332] hover:bg-[#143225] text-white active:scale-95 cursor-pointer shadow-xs'
              : 'bg-stone-100 text-stone-300 cursor-not-allowed'
          }`}
        >
          <Send className="w-4 h-4" />
        </button>
      </form>

    </div>
  );
}
