import React, { useState } from 'react';
import { Download, ArrowLeft, BarChart3, Search, Lightbulb, MessageSquareCode, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import AnalyticsTab from './AnalyticsTab';
import ClauseAnalysisTab from './ClauseAnalysisTab';
import RecommendationsTab from './RecommendationsTab';
import ChatbotTab from './ChatbotTab';

export default function ResultsView({ data, onNewAnalysis, onSendMessage, isThinking, chatHistory, onClearChat }) {
  const [activeTab, setActiveTab] = useState('analytics'); // 'analytics' | 'clauses' | 'recs' | 'chat'

  const {
    filename = 'Analyzed Document',
    total = 0,
    high = 0,
    medium = 0,
    low = 0,
    overall = 'Low',
    risk_score = 0,
    clauses = [],
    report_text = '',
  } = data;

  const handleDownloadReport = () => {
    const blob = new Blob([report_text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'contract_risk_report.txt';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const tabs = [
    { id: 'analytics', label: 'Analytics', icon: BarChart3 },
    { id: 'clauses', label: 'Clause Analysis', icon: Search, badge: total },
    { id: 'recs', label: 'Recommendations', icon: Lightbulb, badge: high + medium },
    { id: 'chat', label: 'AI Chatbot', icon: MessageSquareCode },
  ];

  const overallColor = overall === 'High' ? 'text-rose-600' : overall === 'Medium' ? 'text-amber-600' : 'text-emerald-600';

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      
      {/* Top Header Card (Kombai styled) */}
      <div className="bg-white p-6 rounded-2xl border border-stone-200/90 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        {/* Document Title & Badges */}
        <div className="space-y-1.5">
          <div className="flex items-center space-x-2 text-xs font-mono text-stone-400">
            <span>Audit History</span>
            <span>&gt;</span>
            <span className="text-stone-700 font-semibold">{filename}</span>
          </div>

          <h2 className="text-2xl font-extrabold text-stone-900 tracking-tight">
            {filename}
          </h2>

          <div className="flex flex-wrap items-center gap-3 pt-1 text-xs">
            <span className="text-stone-500 font-mono">
              {total} clauses · analyzed today
            </span>

            <div className="flex items-center space-x-1.5 font-mono text-[11px] font-semibold">
              <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200">
                ● {high} high
              </span>
              <span className="px-2 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200">
                ● {medium} medium
              </span>
              <span className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">
                ● {low} low
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleDownloadReport}
            className="inline-flex items-center space-x-1.5 bg-white hover:bg-stone-50 text-stone-800 border border-stone-300 px-4 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition-all shadow-2xs hover:shadow-xs active:scale-[0.98] cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-stone-600" />
            <span>Export report (.txt)</span>
          </button>

          <button
            onClick={onNewAnalysis}
            className="inline-flex items-center space-x-1.5 bg-[#1b4332] hover:bg-[#143225] text-white px-4 py-2.5 rounded-xl text-xs font-semibold tracking-wide transition-all shadow-xs hover:shadow active:scale-[0.98] cursor-pointer"
          >
            <ArrowLeft className="w-3.5 h-3.5 text-emerald-300" />
            <span>New analysis</span>
          </button>
        </div>

      </div>

      {/* 5 Metric Cards (1:1 with app.py) */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {/* Total Clauses */}
        <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-2xs text-center">
          <div className="text-3xl font-extrabold text-[#1b4332] font-mono">{total}</div>
          <div className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-1 font-semibold">
            Total Clauses
          </div>
        </div>

        {/* Overall Risk */}
        <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-2xs text-center">
          <div className={`text-3xl font-extrabold font-mono ${overallColor}`}>{overall}</div>
          <div className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-1 font-semibold">
            Overall Risk
          </div>
        </div>

        {/* High Risk */}
        <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-2xs text-center">
          <div className="text-3xl font-extrabold text-rose-600 font-mono">{high}</div>
          <div className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-1 font-semibold">
            High Risk
          </div>
        </div>

        {/* Medium Risk */}
        <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-2xs text-center">
          <div className="text-3xl font-extrabold text-amber-600 font-mono">{medium}</div>
          <div className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-1 font-semibold">
            Medium Risk
          </div>
        </div>

        {/* Low Risk */}
        <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-2xs text-center col-span-2 sm:col-span-1">
          <div className="text-3xl font-extrabold text-emerald-600 font-mono">{low}</div>
          <div className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-1 font-semibold">
            Low Risk
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-stone-200 gap-2 sm:gap-4 overflow-x-auto pb-px">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center space-x-2 py-3 px-4 font-semibold text-xs border-b-2 transition-all whitespace-nowrap cursor-pointer ${
                isActive
                  ? 'border-[#1b4332] text-[#1b4332]'
                  : 'border-transparent text-stone-500 hover:text-stone-800 hover:border-stone-300'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-[#1b4332]' : 'text-stone-400'}`} />
              <span>{t.label}</span>
              {t.badge !== undefined && (
                <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded-full ${
                  isActive ? 'bg-emerald-100 text-emerald-900 font-bold' : 'bg-stone-100 text-stone-500'
                }`}>
                  {t.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Tab Content Panels */}
      <div className="min-h-[400px]">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2 }}
          >
            {activeTab === 'analytics' && <AnalyticsTab data={data} />}
            {activeTab === 'clauses' && (
              <ClauseAnalysisTab clauses={clauses} onDownloadReport={handleDownloadReport} />
            )}
            {activeTab === 'recs' && <RecommendationsTab clauses={clauses} />}
            {activeTab === 'chat' && (
              <ChatbotTab
                clauses={clauses}
                onSendMessage={onSendMessage}
                isThinking={isThinking}
                chatHistory={chatHistory}
                onClearChat={onClearChat}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </div>

    </div>
  );
}
