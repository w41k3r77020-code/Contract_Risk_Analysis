import React, { useState } from 'react';
import { Search, Download, Filter, FileText, Lightbulb, AlertCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function ClauseAnalysisTab({ clauses = [], onDownloadReport }) {
  const [filter, setFilter] = useState('All'); // 'All' | 'High' | 'Medium' | 'Low'
  const [searchQuery, setSearchQuery] = useState('');

  // Counts for filter pills
  const counts = {
    All: clauses.length,
    High: clauses.filter(c => c.risk_level === 'High').length,
    Medium: clauses.filter(c => c.risk_level === 'Medium').length,
    Low: clauses.filter(c => c.risk_level === 'Low').length,
  };

  // Filter & Search logic
  const filteredClauses = clauses.filter((item) => {
    const matchesFilter = filter === 'All' || item.risk_level === filter;
    const matchesSearch = searchQuery.trim() === '' || 
      item.clause.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.analysis.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (item.recommendation && item.recommendation.toLowerCase().includes(searchQuery.toLowerCase()));
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="space-y-6">
      
      {/* Header + Filter Bar */}
      <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-4">
        
        {/* Left: Section title */}
        <div>
          <h3 className="text-base font-bold text-stone-900">
            🔍 Clause-by-Clause Breakdown
          </h3>
          <p className="text-xs text-stone-500 mt-0.5">
            Every clause analyzed with risk level, analysis, and recommendation
          </p>
        </div>

        {/* Right: Search + Filter Pills */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-stone-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search clauses..."
              className="pl-8 pr-3 py-1.5 text-xs rounded-lg border border-stone-200 focus:border-emerald-700 outline-none w-44 sm:w-56 font-mono"
            />
          </div>

          {/* Filter Pills (All / High / Med / Low) */}
          <div className="flex items-center space-x-1 bg-stone-100 p-1 rounded-lg text-xs font-mono">
            {['All', 'High', 'Medium', 'Low'].map((cat) => {
              const isActive = filter === cat;
              return (
                <button
                  key={cat}
                  onClick={() => setFilter(cat)}
                  className={`px-2.5 py-1 rounded-md transition-all font-semibold flex items-center space-x-1.5 ${
                    isActive
                      ? 'bg-white text-stone-900 shadow-xs'
                      : 'text-stone-500 hover:text-stone-800'
                  }`}
                >
                  <span>{cat}</span>
                  <span className={`text-[10px] px-1 rounded-full ${
                    cat === 'High' ? 'bg-rose-100 text-rose-700' :
                    cat === 'Medium' ? 'bg-amber-100 text-amber-700' :
                    cat === 'Low' ? 'bg-emerald-100 text-emerald-700' :
                    'bg-stone-200 text-stone-600'
                  }`}>
                    {counts[cat]}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

      </div>

      {/* Clause Cards Grid / List */}
      {filteredClauses.length === 0 ? (
        <div className="bg-white p-12 rounded-2xl border border-stone-200 text-center">
          <AlertCircle className="w-8 h-8 text-stone-400 mx-auto mb-2" />
          <p className="text-sm font-semibold text-stone-700">No matching clauses found</p>
          <p className="text-xs text-stone-400 mt-1">Try switching your filter or clearing search terms.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <AnimatePresence>
            {filteredClauses.map((item) => {
              const originalIndex = clauses.indexOf(item) + 1;
              const isHigh = item.risk_level === 'High';
              const isMed = item.risk_level === 'Medium';

              const badgeColor = isHigh
                ? 'bg-rose-50 text-rose-700 border-rose-200'
                : isMed
                ? 'bg-amber-50 text-amber-700 border-amber-200'
                : 'bg-emerald-50 text-emerald-700 border-emerald-200';

              const cardBorder = isHigh
                ? 'border-l-4 border-l-rose-500 border-stone-200'
                : isMed
                ? 'border-l-4 border-l-amber-500 border-stone-200'
                : 'border-l-4 border-l-emerald-500 border-stone-200';

              return (
                <motion.div
                  key={originalIndex}
                  layout
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className={`bg-white rounded-2xl border ${cardBorder} p-5 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between`}
                >
                  <div className="space-y-3.5">
                    {/* Card Top: Clause # and Risk pill */}
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-400">
                        CLAUSE {originalIndex}
                      </span>
                      <span className={`text-[11px] font-mono font-bold uppercase px-2.5 py-0.5 rounded-full border ${badgeColor}`}>
                        {item.risk_level} Risk
                      </span>
                    </div>

                    {/* Clause Text quote box */}
                    <div className="bg-stone-50/80 p-3.5 rounded-xl border border-stone-200/60 text-xs text-stone-800 leading-relaxed font-serif">
                      "{item.clause}"
                    </div>

                    {/* Analysis Section */}
                    <div className="space-y-1">
                      <div className="flex items-center space-x-1.5 text-xs font-bold text-stone-900">
                        <span>🔍 Analysis:</span>
                      </div>
                      <p className="text-xs text-stone-600 leading-relaxed bg-stone-50/30 p-2 rounded-lg">
                        {item.analysis}
                      </p>
                    </div>

                    {/* Recommendation Section */}
                    <div className="space-y-1">
                      <div className="flex items-center space-x-1.5 text-xs font-bold text-emerald-900">
                        <Lightbulb className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                        <span>Recommendation:</span>
                      </div>
                      <p className="text-xs text-emerald-950 font-medium leading-relaxed bg-emerald-50/60 p-2 rounded-lg border border-emerald-100">
                        {item.recommendation || 'No actionable revision required.'}
                      </p>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      )}

      {/* Download Full Report CTA (1:1 with app.py line 814) */}
      <div className="pt-4">
        <button
          onClick={onDownloadReport}
          className="w-full py-3.5 px-6 rounded-xl bg-stone-900 hover:bg-stone-800 text-white font-semibold text-xs tracking-wide uppercase transition-all shadow-sm flex items-center justify-center space-x-2 active:scale-[0.99] cursor-pointer"
        >
          <Download className="w-4 h-4" />
          <span>⬇️ Download Full Report (.txt)</span>
        </button>
      </div>

    </div>
  );
}
