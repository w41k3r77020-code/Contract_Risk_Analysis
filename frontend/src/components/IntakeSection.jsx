import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, Lock, CheckCircle, Loader2, AlertCircle, X, RefreshCw } from 'lucide-react';
import { motion } from 'framer-motion';

export default function IntakeSection({ onStartAnalysis, isLoading, currentStep, error }) {
  const [activeTab, setActiveTab] = useState('pdf'); // 'pdf' | 'text'
  const [selectedFile, setSelectedFile] = useState(null);
  const [pastedText, setPastedText] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const isTextValid = pastedText.trim().length >= 50;
  const canSubmit = !isLoading && ((activeTab === 'pdf' && selectedFile) || (activeTab === 'text' && isTextValid));

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canSubmit) return;

    if (activeTab === 'pdf') {
      onStartAnalysis({ file: selectedFile, text: null });
    } else {
      onStartAnalysis({ file: null, text: pastedText });
    }
  };

  // Steps matching the Kombai design (1:1 with agent workflow)
  const steps = [
    { num: 1, title: 'Parsing', desc: 'Separating contract clauses' },
    { num: 2, title: 'Retrieving context', desc: 'Finding relevant supporting material' },
    { num: 3, title: 'Analyzing', desc: 'Assessing clause risk' },
    { num: 4, title: 'Report ready', desc: 'Preparing the clause register' },
  ];

  return (
    <section id="intake-section" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 scroll-mt-20">
      <div className="border-t border-stone-200/80 pt-10">
        
        {/* Section Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-2 text-xs font-mono font-medium text-stone-500 uppercase tracking-wider">
            <span>02 / INTAKE</span>
            <span>—</span>
            <span>NEW ANALYSIS</span>
          </div>
          <h2 className="text-3xl font-extrabold text-stone-900 tracking-tight mt-1">
            New analysis
          </h2>
          <p className="text-sm text-stone-500 mt-1">
            Upload a PDF or paste contract text for clause-level review.
          </p>
        </div>

        {/* Error Notification */}
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-start space-x-3"
          >
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Analysis Failed</p>
              <p className="text-rose-700 text-xs mt-0.5">{error}</p>
            </div>
          </motion.div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Contract Source */}
          <div className="lg:col-span-7 bg-white rounded-2xl border border-stone-200/90 shadow-sm p-6 sm:p-7 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-stone-100 mb-5">
                <span className="text-xs font-mono uppercase font-bold text-stone-800 tracking-wider">
                  Contract source
                </span>
                <span className="text-xs text-stone-400 font-mono">Secure workspace upload</span>
              </div>

              {/* Source Switcher Tabs */}
              <div className="flex space-x-1 bg-stone-100 p-1 rounded-xl mb-6">
                <button
                  type="button"
                  onClick={() => setActiveTab('pdf')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all flex items-center justify-center space-x-2 ${
                    activeTab === 'pdf'
                      ? 'bg-white text-stone-900 shadow-sm'
                      : 'text-stone-500 hover:text-stone-800'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Upload PDF</span>
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('text')}
                  className={`flex-1 py-2 text-xs font-semibold rounded-lg transition-all flex items-center justify-center space-x-2 ${
                    activeTab === 'text'
                      ? 'bg-white text-stone-900 shadow-sm'
                      : 'text-stone-500 hover:text-stone-800'
                  }`}
                >
                  <span>✏️</span>
                  <span>Paste text</span>
                </button>
              </div>

              {/* Tab 1: PDF Upload */}
              {activeTab === 'pdf' && (
                <div>
                  {!selectedFile ? (
                    <div
                      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                      onDragLeave={() => setIsDragOver(false)}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all ${
                        isDragOver
                          ? 'border-emerald-600 bg-emerald-50/50'
                          : 'border-stone-300 hover:border-emerald-600/70 hover:bg-stone-50/50'
                      }`}
                    >
                      <input
                        ref={fileInputRef}
                        type="file"
                        accept="application/pdf"
                        onChange={handleFileChange}
                        className="hidden"
                      />
                      <div className="w-12 h-12 rounded-full bg-stone-100 text-stone-600 mx-auto flex items-center justify-center mb-4 transition-transform group-hover:scale-110">
                        <UploadCloud className="w-6 h-6 text-emerald-800" />
                      </div>
                      <p className="text-sm font-semibold text-stone-800">
                        Drop a PDF here <span className="font-normal text-stone-500">or choose a file</span>
                      </p>
                      <p className="text-xs text-stone-400 mt-1.5 font-mono">
                        Text-based PDFs supported · max 50 MB
                      </p>
                    </div>
                  ) : (
                    <div className="p-4 bg-stone-50 rounded-xl border border-stone-200 flex items-center justify-between">
                      <div className="flex items-center space-x-3 truncate">
                        <div className="w-10 h-10 rounded-lg bg-emerald-100/70 text-emerald-800 flex items-center justify-center shrink-0">
                          <FileText className="w-5 h-5" />
                        </div>
                        <div className="truncate">
                          <p className="text-xs font-bold text-stone-900 truncate">{selectedFile.name}</p>
                          <p className="text-[11px] text-stone-500 font-mono mt-0.5">
                            {(selectedFile.size / 1024).toFixed(1)} KB · Ready to analyze
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2 shrink-0 ml-4">
                        <button
                          type="button"
                          onClick={() => fileInputRef.current?.click()}
                          className="text-xs font-semibold text-stone-600 hover:text-stone-900 px-2 py-1"
                        >
                          Replace
                        </button>
                        <button
                          type="button"
                          onClick={handleRemoveFile}
                          className="p-1 rounded-md text-stone-400 hover:text-rose-600 hover:bg-rose-50 transition-colors"
                          title="Remove file"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Paste Text */}
              {activeTab === 'text' && (
                <div className="space-y-2">
                  <textarea
                    rows={8}
                    value={pastedText}
                    onChange={(e) => setPastedText(e.target.value)}
                    placeholder="Paste contract clauses or entire agreement here... (e.g., 1. Definitions... 2. Limitation of Liability...)"
                    className="w-full text-xs font-mono p-4 rounded-xl border border-stone-300 focus:border-emerald-700 focus:ring-1 focus:ring-emerald-700 outline-none transition-all placeholder:text-stone-400 bg-stone-50/50"
                  />
                  <div className="flex justify-between items-center text-xs font-mono text-stone-500">
                    <span className={pastedText.length > 0 && pastedText.length < 50 ? 'text-amber-600' : ''}>
                      {pastedText.length} characters (min 50 required)
                    </span>
                    {pastedText.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setPastedText('')}
                        className="text-stone-400 hover:text-stone-600"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Security Notice */}
            <div className="pt-6 mt-6 border-t border-stone-100 flex items-center space-x-2 text-xs text-stone-400">
              <Lock className="w-3.5 h-3.5 shrink-0" />
              <span>Input stays with this analysis while the report is generated.</span>
            </div>
          </div>

          {/* Right Column: Analysis Progress */}
          <div className="lg:col-span-5 bg-white rounded-2xl border border-stone-200/90 shadow-sm p-6 sm:p-7 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-stone-100 mb-6">
                <span className="text-xs font-mono uppercase font-bold text-stone-800 tracking-wider">
                  Analysis progress
                </span>
                <span className="text-xs font-mono text-stone-400">
                  {isLoading ? 'In progress...' : 'Awaiting start'}
                </span>
              </div>

              {/* Step list */}
              <div className="space-y-6">
                {steps.map((step) => {
                  const isCompleted = currentStep > step.num;
                  const isCurrent = currentStep === step.num && isLoading;
                  const isPending = currentStep < step.num;

                  return (
                    <div key={step.num} className="flex items-start space-x-4">
                      {/* Step Indicator Circle */}
                      <div className="shrink-0 mt-0.5">
                        {isCompleted ? (
                          <div className="w-7 h-7 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center font-mono font-bold text-xs">
                            <CheckCircle className="w-4 h-4" />
                          </div>
                        ) : isCurrent ? (
                          <div className="w-7 h-7 rounded-full bg-[#1b4332] text-white flex items-center justify-center font-mono font-bold text-xs shadow-sm ring-4 ring-emerald-100">
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          </div>
                        ) : (
                          <div className="w-7 h-7 rounded-full border border-stone-300 text-stone-400 flex items-center justify-center font-mono font-bold text-xs">
                            {step.num}
                          </div>
                        )}
                      </div>

                      {/* Step Info */}
                      <div>
                        <p className={`text-xs font-bold ${
                          isCurrent ? 'text-stone-900' : isCompleted ? 'text-emerald-900' : 'text-stone-500'
                        }`}>
                          {step.title}
                        </p>
                        <p className="text-[11px] text-stone-400 mt-0.5">
                          {step.desc}
                        </p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Bottom CTA Action Button */}
            <div className="pt-8 border-t border-stone-100 mt-8">
              <button
                type="button"
                onClick={handleSubmit}
                disabled={!canSubmit}
                className={`w-full py-3.5 px-5 rounded-xl font-semibold text-xs tracking-wide uppercase transition-all flex items-center justify-center space-x-2 shadow-sm ${
                  canSubmit
                    ? 'bg-[#1b4332] hover:bg-[#143225] text-white active:scale-[0.99] cursor-pointer'
                    : 'bg-stone-200 text-stone-400 cursor-not-allowed'
                }`}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin text-emerald-300" />
                    <span>Analyzing clauses with agent...</span>
                  </>
                ) : (
                  <>
                    <FileText className="w-4 h-4" />
                    <span>Start analysis</span>
                  </>
                )}
              </button>

              <p className="text-[11px] text-stone-400 text-center mt-2.5">
                AI-generated insights, not legal advice.
              </p>
            </div>

          </div>

        </div>

      </div>
    </section>
  );
}
