import React from 'react';
import { Lightbulb, AlertTriangle, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

export default function RecommendationsTab({ clauses = [] }) {
  // Sort clauses by High (0), Medium (1), Low (2)
  const indexedClauses = clauses.map((c, i) => ({ ...c, originalIndex: i + 1 }));

  const groups = [
    {
      key: 'High',
      title: '🔴 High Priority — Immediate Action Required',
      subtitle: 'Critical liability or compliance risks that should be negotiated before signing',
      borderColor: 'border-l-rose-500',
      badgeClass: 'bg-rose-50 text-rose-700 border-rose-200',
      items: indexedClauses.filter(c => c.risk_level === 'High'),
      icon: ShieldAlert,
    },
    {
      key: 'Medium',
      title: '🟡 Medium Priority — Review Suggested',
      subtitle: 'Ambiguous or imbalanced terms that warrant legal counsel clarification',
      borderColor: 'border-l-amber-500',
      badgeClass: 'bg-amber-50 text-amber-700 border-amber-200',
      items: indexedClauses.filter(c => c.risk_level === 'Medium'),
      icon: AlertTriangle,
    },
    {
      key: 'Low',
      title: '🟢 Low Priority — Acceptable',
      subtitle: 'Standard commercial clauses conforming to standard practice',
      borderColor: 'border-l-emerald-500',
      badgeClass: 'bg-emerald-50 text-emerald-700 border-emerald-200',
      items: indexedClauses.filter(c => c.risk_level === 'Low'),
      icon: CheckCircle2,
    },
  ];

  return (
    <div className="space-y-8">
      
      {/* Header */}
      <div className="bg-white p-5 rounded-2xl border border-stone-200/90 shadow-sm">
        <h3 className="text-base font-bold text-stone-900">
          💡 AI-Powered Recommendations
        </h3>
        <p className="text-xs text-stone-500 mt-0.5">
          Actionable suggestions prioritized by risk severity
        </p>
      </div>

      {/* Groups */}
      {groups.map((group) => {
        if (group.items.length === 0) return null;

        return (
          <div key={group.key} className="space-y-4">
            {/* Group Header */}
            <div>
              <h4 className="text-sm font-bold text-stone-900 flex items-center space-x-2">
                <span>{group.title}</span>
                <span className="text-xs font-mono font-normal text-stone-400">
                  ({group.items.length} {group.items.length === 1 ? 'clause' : 'clauses'})
                </span>
              </h4>
              <p className="text-xs text-stone-500 mt-0.5">
                {group.subtitle}
              </p>
            </div>

            {/* List of cards in this group */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {group.items.map((item) => {
                const preview = item.clause.length > 180 
                  ? item.clause.slice(0, 180) + '...' 
                  : item.clause;

                return (
                  <motion.div
                    key={item.originalIndex}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`bg-white rounded-2xl border border-stone-200/90 border-l-4 ${group.borderColor} p-5 shadow-xs hover:shadow-md transition-shadow flex flex-col justify-between space-y-3`}
                  >
                    <div>
                      {/* Top bar */}
                      <div className="flex items-center justify-between pb-2 border-b border-stone-100">
                        <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-400">
                          Clause {item.originalIndex}
                        </span>
                        <span className={`text-[11px] font-mono font-bold uppercase px-2 py-0.5 rounded-full border ${group.badgeClass}`}>
                          {item.risk_level} Risk
                        </span>
                      </div>

                      {/* Snippet */}
                      <p className="text-xs text-stone-600 mt-2.5 font-serif italic line-clamp-3">
                        "{preview}"
                      </p>
                    </div>

                    {/* Recommendation Box */}
                    <div className="p-3 bg-emerald-50/50 rounded-xl border border-emerald-200/60 text-xs text-emerald-950 font-medium">
                      <div className="flex items-center space-x-1.5 font-bold text-emerald-900 mb-1">
                        <Lightbulb className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                        <span>Suggested Fix / Recommendation:</span>
                      </div>
                      <p className="leading-relaxed">
                        {item.recommendation || 'No specific action required.'}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        );
      })}

    </div>
  );
}
