import React from 'react';
import { FileText, ArrowRight, ShieldCheck, CheckCircle2, ChevronRight, Scale, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

export default function LandingHero({ onScrollToIntake }) {
  return (
    <section className="relative overflow-hidden pt-8 pb-16">
      {/* Background Subtle Gradient Blobs */}
      <div className="absolute top-0 right-1/4 w-96 h-96 bg-emerald-100/40 rounded-full blur-3xl -z-10 pointer-events-none" />
      <div className="absolute top-1/3 left-10 w-80 h-80 bg-stone-200/50 rounded-full blur-3xl -z-10 pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          
          {/* Left Hero Column */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="lg:col-span-6 space-y-6"
          >
            <div className="inline-flex items-center space-x-2 text-xs font-mono font-semibold tracking-wider text-emerald-800 uppercase">
              <span className="h-px w-6 bg-emerald-800"></span>
              <span>CONTRACT RISK ANALYSIS</span>
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-stone-900 tracking-tight leading-[1.1]">
              Turn a contract into a <span className="underline decoration-emerald-600/30 underline-offset-8">clause-level</span> risk report.
            </h1>

            <p className="text-base sm:text-lg text-stone-600 leading-relaxed max-w-xl">
              Upload a PDF or paste contract text. ClauseGuard identifies High, Medium, and Low risk clauses, explains the result with Groq LLM reasoning, and suggests actionable next steps.
            </p>

            <div className="pt-2 flex flex-col sm:flex-row items-start sm:items-center space-y-3 sm:space-y-0 sm:space-x-4">
              <button
                onClick={onScrollToIntake}
                className="inline-flex items-center space-x-2 bg-[#1b4332] hover:bg-[#143225] text-white px-6 py-3.5 rounded-xl font-semibold text-sm shadow-md hover:shadow-lg transition-all active:scale-[0.98]"
              >
                <FileText className="w-4 h-4 text-emerald-300" />
                <span>Analyze a contract</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>

              <div className="flex items-center space-x-2 text-xs text-stone-500 font-mono">
                <span className="w-1.5 h-1.5 rounded-full bg-stone-400"></span>
                <span>PDF or pasted text · min 50 characters</span>
              </div>
            </div>

            {/* Feature bullets */}
            <div className="grid grid-cols-3 gap-3 pt-6 border-t border-stone-200/80">
              <div className="space-y-1">
                <p className="text-xs font-mono font-semibold text-stone-400">01</p>
                <p className="text-xs font-semibold text-stone-800">Risk Level per Clause</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-mono font-semibold text-stone-400">02</p>
                <p className="text-xs font-semibold text-stone-800">Reasoned Explanations</p>
              </div>
              <div className="space-y-1">
                <p className="text-xs font-mono font-semibold text-stone-400">03</p>
                <p className="text-xs font-semibold text-stone-800">Actionable Fixes</p>
              </div>
            </div>
          </motion.div>

          {/* Right Illustrative Card (Sample UI Preview from Kombai design) */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.15 }}
            className="lg:col-span-6"
          >
            <div className="bg-white rounded-2xl border border-stone-200/90 shadow-xl overflow-hidden">
              {/* Card topbar */}
              <div className="px-6 py-3.5 border-b border-stone-100 flex items-center justify-between bg-stone-50/50">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-mono uppercase tracking-wider text-stone-400 font-medium">ILLUSTRATIVE ANALYSIS OUTPUT</span>
                </div>
                <span className="px-2 py-0.5 text-[10px] font-mono uppercase font-bold tracking-wider bg-stone-200/70 text-stone-700 rounded">
                  SAMPLE UI
                </span>
              </div>

              <div className="p-6 space-y-6">
                {/* Contract title */}
                <div>
                  <h3 className="text-lg font-bold text-stone-900">Master Services Agreement</h3>
                  <p className="text-xs text-stone-500 mt-0.5">Generated from uploaded contract · 32 clauses reviewed</p>
                </div>

                {/* Risk distribution segmented bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-stone-500 uppercase tracking-wider font-semibold">RISK DISTRIBUTION</span>
                    <span className="text-stone-700 font-bold">32 TOTAL</span>
                  </div>
                  <div className="h-3 w-full flex rounded-full overflow-hidden bg-stone-100">
                    <div style={{ width: '15%' }} className="bg-rose-500 transition-all" title="High: 2" />
                    <div style={{ width: '25%' }} className="bg-amber-400 transition-all" title="Medium: 5" />
                    <div style={{ width: '60%' }} className="bg-emerald-600 transition-all" title="Low: 25" />
                  </div>
                  <div className="flex items-center space-x-4 text-xs font-mono">
                    <span className="text-rose-600 font-medium">High 2</span>
                    <span className="text-amber-600 font-medium">Medium 5</span>
                    <span className="text-emerald-700 font-medium">Low 25</span>
                  </div>
                </div>

                {/* Sample Clause Rows */}
                <div className="divide-y divide-stone-100 border-t border-stone-100 pt-2">
                  <div className="py-3 flex items-start justify-between group hover:bg-stone-50/50 px-2 rounded-lg transition-colors">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-stone-900">Limitation of liability</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-50 text-rose-700 font-semibold border border-rose-200">HIGH</span>
                      </div>
                      <p className="text-xs text-stone-500 mt-1">Cap may not cover data protection obligations.</p>
                    </div>
                    <span className="text-xs font-mono text-stone-400 ml-4 whitespace-nowrap">§ 14.2</span>
                  </div>

                  <div className="py-3 flex items-start justify-between group hover:bg-stone-50/50 px-2 rounded-lg transition-colors">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-stone-900">Confidentiality</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 font-semibold border border-amber-200">MEDIUM</span>
                      </div>
                      <p className="text-xs text-stone-500 mt-1">Survival period may be short for sensitive intellectual property.</p>
                    </div>
                    <span className="text-xs font-mono text-stone-400 ml-4 whitespace-nowrap">§ 7.1</span>
                  </div>

                  <div className="py-3 flex items-start justify-between group hover:bg-stone-50/50 px-2 rounded-lg transition-colors">
                    <div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs font-bold text-stone-900">Termination for Convenience</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 font-semibold border border-emerald-200">LOW</span>
                      </div>
                      <p className="text-xs text-stone-500 mt-1">Clear 30-day notice requirement and standard terms.</p>
                    </div>
                    <span className="text-xs font-mono text-stone-400 ml-4 whitespace-nowrap">§ 19.1</span>
                  </div>
                </div>

                {/* Card footer note */}
                <div className="p-3 bg-stone-50 rounded-xl border border-stone-200/60 flex items-center space-x-2.5 text-xs text-stone-600">
                  <Scale className="w-4 h-4 text-emerald-700 shrink-0" />
                  <span>Interactive clause findings with instant legal context retrieval</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Value Prop Banner (Section 2 from Kombai design) */}
        <div className="mt-16 pt-12 border-t border-stone-200/80">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-8 items-center">
            <div className="md:col-span-7 space-y-3">
              <div className="text-xs font-mono font-semibold tracking-wider text-emerald-800 uppercase">
                CAPABILITY BOUNDARY
              </div>
              <h2 className="text-2xl font-bold text-stone-900">
                What ClauseGuard gives you.
              </h2>
              <p className="text-stone-600 text-sm">
                A focused analysis output for deciding where legal attention should go next.
              </p>
            </div>

            <div className="md:col-span-5 bg-white p-5 rounded-xl border border-stone-200 shadow-sm flex items-start space-x-3.5">
              <div className="p-2 bg-emerald-50 rounded-lg text-emerald-700 shrink-0">
                <CheckCircle2 className="w-5 h-5" />
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-bold text-stone-900">Human review stays in the loop</h4>
                <p className="text-xs text-stone-500 leading-relaxed">
                  Use the report to focus legal attention; it provides AI-generated assistance and does not replace qualified legal counsel.
                </p>
              </div>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
