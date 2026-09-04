import React from 'react';
import { ShieldCheck, Scale, Cpu, Sparkles } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-stone-200/90 bg-[#faf9f6] py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Top footer disclaimer */}
        <div className="p-4 rounded-xl bg-amber-50/70 border border-amber-200/70 text-amber-900 text-xs text-center flex items-center justify-center space-x-2 mb-8">
          <Scale className="w-4 h-4 text-amber-700 shrink-0" />
          <span>
            <strong>AI-Generated Insights, Not Legal Advice.</strong> Review findings with qualified legal counsel before taking contractual action.
          </span>
        </div>

        {/* Bottom row */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-stone-500 font-mono">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded bg-[#1b4332] text-white flex items-center justify-center">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-300" />
            </div>
            <span className="font-bold text-stone-800">ClauseGuard</span>
            <span>—</span>
            <span>PDF or pasted text → clause-level risk analysis.</span>
          </div>

          <div className="flex items-center space-x-4">
            <span className="flex items-center space-x-1 text-emerald-800">
              <Cpu className="w-3.5 h-3.5" />
              <span>SentenceTransformers + FAISS + Groq</span>
            </span>
          </div>
        </div>

      </div>
    </footer>
  );
}
