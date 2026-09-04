import React from 'react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { ShieldAlert, AlertTriangle, CheckCircle, Activity } from 'lucide-react';

export default function AnalyticsTab({ data }) {
  const { total = 0, high = 0, medium = 0, low = 0, overall = 'Low', risk_score = 0 } = data;

  const pieData = [
    { name: 'High Risk', value: high, color: '#ef4444' },
    { name: 'Medium Risk', value: medium, color: '#f59e0b' },
    { name: 'Low Risk', value: low, color: '#10b981' },
  ].filter(d => d.value > 0);

  const barData = [
    { name: 'High', count: high, fill: '#ef4444' },
    { name: 'Medium', count: medium, fill: '#f59e0b' },
    { name: 'Low', count: low, fill: '#10b981' },
  ];

  const gaugeColor = risk_score >= 60 ? '#ef4444' : risk_score >= 30 ? '#f59e0b' : '#10b981';

  // SVG Gauge calculations (semi-circle)
  const radius = 80;
  const strokeWidth = 14;
  const circumference = Math.PI * radius; // Half-circle circumference
  const strokeDashoffset = circumference - (risk_score / 100) * circumference;

  return (
    <div className="space-y-6">
      
      {/* Top Row: Gauge + Summary Table */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Gauge Card */}
        <div className="lg:col-span-5 bg-white p-6 rounded-2xl border border-stone-200/90 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-700 flex items-center space-x-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-700" />
              <span>Overall Risk Score</span>
            </span>
            <span className={`text-[11px] font-mono font-bold px-2 py-0.5 rounded uppercase ${
              risk_score >= 60 ? 'bg-rose-50 text-rose-700 border border-rose-200' :
              risk_score >= 30 ? 'bg-amber-50 text-amber-700 border border-amber-200' :
              'bg-emerald-50 text-emerald-700 border border-emerald-200'
            }`}>
              {overall} Risk
            </span>
          </div>

          <div className="py-6 flex flex-col items-center justify-center">
            {/* SVG Meter */}
            <div className="relative w-48 h-28 flex items-center justify-center">
              <svg className="w-48 h-28 overflow-visible" viewBox="0 0 200 120">
                {/* Background Arc */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke="#f5f5f4"
                  strokeWidth={strokeWidth}
                  strokeLinecap="round"
                />
                {/* Value Arc */}
                <path
                  d="M 20 100 A 80 80 0 0 1 180 100"
                  fill="none"
                  stroke={gaugeColor}
                  strokeWidth={strokeWidth}
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="round"
                  className="transition-all duration-1000 ease-out"
                />
              </svg>

              {/* Center Text */}
              <div className="absolute bottom-2 flex flex-col items-center">
                <span className="text-3xl font-extrabold text-stone-900 tracking-tight">
                  {risk_score}%
                </span>
                <span className="text-[11px] font-mono text-stone-400 uppercase tracking-wider mt-0.5">
                  composite score
                </span>
              </div>
            </div>

            {/* Threshold Legend */}
            <div className="flex items-center space-x-4 text-[11px] font-mono text-stone-500 mt-4">
              <span className="flex items-center space-x-1">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                <span>0-29% Low</span>
              </span>
              <span className="flex items-center space-x-1">
                <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                <span>30-59% Med</span>
              </span>
              <span className="flex items-center space-x-1">
                <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                <span>60%+ High</span>
              </span>
            </div>
          </div>

          <p className="text-xs text-stone-400 text-center border-t border-stone-100 pt-3">
            Weighted calculation across all flagged clause liabilities.
          </p>
        </div>

        {/* Quick Summary Table Card */}
        <div className="lg:col-span-7 bg-white p-6 rounded-2xl border border-stone-200/90 shadow-sm flex flex-col justify-between">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-700">
              📋 Quick Summary
            </span>
            <span className="text-xs font-mono text-stone-400">Clause Register Audit</span>
          </div>

          <div className="overflow-x-auto my-3">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-stone-100 text-[11px] font-mono uppercase text-stone-400">
                  <th className="pb-2.5 font-semibold">Metric</th>
                  <th className="pb-2.5 font-semibold">Count / Value</th>
                  <th className="pb-2.5 font-semibold text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-stone-100 text-xs">
                <tr>
                  <td className="py-3 text-stone-700 font-medium">Total Clauses Analyzed</td>
                  <td className="py-3 font-mono font-bold text-stone-900">{total}</td>
                  <td className="py-3 text-right">
                    <span className="inline-flex items-center space-x-1 text-emerald-700 font-medium text-[11px]">
                      <CheckCircle className="w-3.5 h-3.5" />
                      <span>Complete</span>
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 text-stone-700 font-medium">High Risk Clauses</td>
                  <td className="py-3 font-mono font-bold text-rose-600">{high}</td>
                  <td className="py-3 text-right">
                    {high > 0 ? (
                      <span className="inline-flex items-center space-x-1 text-rose-700 font-medium text-[11px]">
                        <ShieldAlert className="w-3.5 h-3.5" />
                        <span>Action Required</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 text-emerald-700 font-medium text-[11px]">
                        <CheckCircle className="w-3.5 h-3.5" />
                        <span>Clear</span>
                      </span>
                    )}
                  </td>
                </tr>

                <tr>
                  <td className="py-3 text-stone-700 font-medium">Medium Risk Clauses</td>
                  <td className="py-3 font-mono font-bold text-amber-600">{medium}</td>
                  <td className="py-3 text-right">
                    {medium > 0 ? (
                      <span className="inline-flex items-center space-x-1 text-amber-700 font-medium text-[11px]">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        <span>Review Suggested</span>
                      </span>
                    ) : (
                      <span className="inline-flex items-center space-x-1 text-emerald-700 font-medium text-[11px]">
                        <CheckCircle className="w-3.5 h-3.5" />
                        <span>Clear</span>
                      </span>
                    )}
                  </td>
                </tr>

                <tr>
                  <td className="py-3 text-stone-700 font-medium">Low Risk Clauses</td>
                  <td className="py-3 font-mono font-bold text-emerald-600">{low}</td>
                  <td className="py-3 text-right">
                    <span className="inline-flex items-center space-x-1 text-emerald-700 font-medium text-[11px]">
                      <CheckCircle className="w-3.5 h-3.5" />
                      <span>Acceptable</span>
                    </span>
                  </td>
                </tr>

                <tr className="bg-stone-50/60 font-semibold">
                  <td className="py-3 pl-2 text-stone-900">Overall Risk Rating</td>
                  <td className="py-3 font-mono" style={{ color: gaugeColor }}>{risk_score}%</td>
                  <td className="py-3 pr-2 text-right">
                    <span className="font-mono text-xs font-bold" style={{ color: gaugeColor }}>
                      {overall} Risk
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="text-[11px] text-stone-400 pt-2 border-t border-stone-100 flex justify-between">
            <span>Audit timestamp: Today</span>
            <span>RAG context matched: 3-NN</span>
          </div>
        </div>

      </div>

      {/* Bottom Row: 2 Recharts Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* Pie / Donut Chart */}
        <div className="bg-white p-6 rounded-2xl border border-stone-200/90 shadow-sm">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100 mb-4">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-700">
              🍩 Risk Distribution
            </span>
            <span className="text-xs font-mono text-stone-400">{total} clauses</span>
          </div>

          <div className="h-64 w-full flex items-center justify-center">
            {total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1c1917',
                      borderColor: '#44403c',
                      borderRadius: '8px',
                      color: '#fafaf9',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <p className="text-xs text-stone-400">No clause distribution data</p>
            )}
          </div>

          <div className="flex justify-center space-x-6 text-xs font-mono pt-2 border-t border-stone-100">
            <span className="flex items-center space-x-1.5 text-rose-600 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
              <span>High ({high})</span>
            </span>
            <span className="flex items-center space-x-1.5 text-amber-600 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
              <span>Medium ({medium})</span>
            </span>
            <span className="flex items-center space-x-1.5 text-emerald-600 font-semibold">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
              <span>Low ({low})</span>
            </span>
          </div>
        </div>

        {/* Bar Breakdown Chart */}
        <div className="bg-white p-6 rounded-2xl border border-stone-200/90 shadow-sm">
          <div className="flex items-center justify-between pb-3 border-b border-stone-100 mb-4">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-stone-700">
              📊 Risk Count Breakdown
            </span>
            <span className="text-xs font-mono text-stone-400">Severity Totals</span>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData} margin={{ top: 20, right: 20, left: -10, bottom: 5 }}>
                <XAxis dataKey="name" stroke="#a8a29e" fontSize={12} tickLine={false} />
                <YAxis allowDecimals={false} stroke="#a8a29e" fontSize={12} tickLine={false} />
                <Tooltip
                  cursor={{ fill: 'rgba(0, 0, 0, 0.04)' }}
                  contentStyle={{
                    backgroundColor: '#1c1917',
                    borderColor: '#44403c',
                    borderRadius: '8px',
                    color: '#fafaf9',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={`bar-cell-${index}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="text-[11px] text-stone-400 text-center pt-2 border-t border-stone-100">
            Discrete distribution of identified contractual risks
          </div>
        </div>

      </div>

    </div>
  );
}
