import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LandingHero from './components/LandingHero';
import IntakeSection from './components/IntakeSection';
import ResultsView from './components/ResultsView';
import Footer from './components/Footer';
import { analyzeContract, chatWithContract } from './services/api';

export default function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [error, setError] = useState(null);

  // Chatbot state
  const [chatHistory, setChatHistory] = useState([]);
  const [isThinking, setIsThinking] = useState(false);

  // Scroll helper
  const handleScrollToIntake = () => {
    const el = document.getElementById('intake-section');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // Start analysis handler
  const handleStartAnalysis = async ({ file, text }) => {
    setError(null);
    setIsLoading(true);
    setCurrentStep(1);

    // Dynamic progression of visual steps while backend processes
    const step2Timer = setTimeout(() => setCurrentStep(2), 1200);
    const step3Timer = setTimeout(() => setCurrentStep(3), 3200);

    try {
      const result = await analyzeContract({ file, text });
      setCurrentStep(4);
      
      // Smooth transition to results
      setTimeout(() => {
        setAnalysisData(result);
        setIsLoading(false);
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }, 600);

    } catch (err) {
      clearTimeout(step2Timer);
      clearTimeout(step3Timer);
      setIsLoading(false);
      setCurrentStep(1);
      setError(err.message || 'An unexpected error occurred during analysis.');
    }
  };

  // Reset to landing/intake
  const handleNewAnalysis = () => {
    setAnalysisData(null);
    setChatHistory([]);
    setError(null);
    setCurrentStep(1);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Chat message handler
  const handleSendMessage = async (question) => {
    if (!analysisData?.clauses) return;

    // Append user query
    setChatHistory((prev) => [...prev, { role: 'user', content: question }]);
    setIsThinking(true);

    try {
      const response = await chatWithContract({
        question,
        clauses: analysisData.clauses,
      });

      setChatHistory((prev) => [
        ...prev,
        { role: 'assistant', content: response.answer || 'No response returned.' },
      ]);
    } catch (err) {
      setChatHistory((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${err.message}`,
        },
      ]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleClearChat = () => {
    setChatHistory([]);
  };

  return (
    <div className="min-h-screen flex flex-col bg-[#faf9f6] text-stone-900 selection:bg-emerald-100 selection:text-emerald-900">
      <Navbar onNewAnalysis={handleNewAnalysis} hasData={!!analysisData} />

      <main className="flex-1">
        {!analysisData ? (
          <>
            <LandingHero onScrollToIntake={handleScrollToIntake} />
            <IntakeSection
              onStartAnalysis={handleStartAnalysis}
              isLoading={isLoading}
              currentStep={currentStep}
              error={error}
            />
          </>
        ) : (
          <ResultsView
            data={analysisData}
            onNewAnalysis={handleNewAnalysis}
            onSendMessage={handleSendMessage}
            isThinking={isThinking}
            chatHistory={chatHistory}
            onClearChat={handleClearChat}
          />
        )}
      </main>

      <Footer />
    </div>
  );
}
