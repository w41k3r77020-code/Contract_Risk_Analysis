import React from 'react';
import { ShieldCheck, ArrowRight, FileText } from 'lucide-react';

export default function Navbar({ onNewAnalysis, hasData }) {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-stone-200/80 bg-[#faf9f6]/90 backdrop-blur-md transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <div className="flex items-center space-x-3 cursor-pointer" onClick={onNewAnalysis}>
          <div className="w-9 h-9 rounded-lg bg-[#1b4332] flex items-center justify-center text-white shadow-sm transition-transform hover:scale-105">
            <ShieldCheck className="w-5 h-5 text-emerald-300" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="font-bold text-lg tracking-tight text-stone-900">ClauseGuard</span>
            <span className="hidden sm:inline-block text-xs uppercase tracking-wider text-stone-500 font-mono pl-2 border-l border-stone-300">
              Contract Risk Analysis
            </span>
          </div>
        </div>

        {/* Right CTA */}
        <div className="flex items-center space-x-3">
          <div className="hidden md:flex items-center space-x-2 text-xs text-stone-500 bg-stone-100/80 px-2.5 py-1 rounded-full border border-stone-200">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span>RAG + LangGraph Engine</span>
          </div>
          <button
            onClick={() => {
              if (hasData) {
                onNewAnalysis();
              } else {
                document.getElementById('intake-section')?.scrollIntoView({ behavior: 'smooth' });
              }
            }}
            className="inline-flex items-center space-x-1.5 bg-[#1b4332] hover:bg-[#143225] text-white px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all shadow-sm hover:shadow active:scale-[0.98]"
          >
            <FileText className="w-3.5 h-3.5" />
            <span>{hasData ? 'New Analysis' : 'Analyze a contract'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
